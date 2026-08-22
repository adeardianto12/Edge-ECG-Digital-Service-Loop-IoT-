"""Evaluate the frozen PVC model on INCART without training or tuning.

The script intentionally has no training, calibration-fitting, or threshold-
selection path. It uses lead II, resamples every record to 360 Hz, and keeps
only N and V annotations so that the external task matches the frozen model.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import tensorflow as tf
import wfdb
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


WINDOW_SIZE = 300
HALF_WINDOW = WINDOW_SIZE // 2
TARGET_SAMPLE_RATE_HZ = 360
LEAD_NAME = "II"
RHYTHM_HISTORY_SIZE = 8
BEAT_SYMBOLS = {"N", "V"}
R_PEAK_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j",
    "n", "E", "/", "f", "Q", "?", "P", "U",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalise_beat(beat: np.ndarray) -> np.ndarray:
    std = beat.std()
    return (beat - beat.mean()) / (std if std > 1e-8 else 1.0)


def resample_signal(signal: np.ndarray, source_hz: float) -> np.ndarray:
    ratio = Fraction(TARGET_SAMPLE_RATE_HZ / source_hz).limit_denominator(10000)
    return resample_poly(signal, ratio.numerator, ratio.denominator).astype(np.float32)


def extract_record(record_base: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record = wfdb.rdrecord(str(record_base))
    annotation = wfdb.rdann(str(record_base), "atr")
    if LEAD_NAME not in record.sig_name:
        raise ValueError(f"{record_base.name} has no fixed external-test lead {LEAD_NAME}")
    lead_index = record.sig_name.index(LEAD_NAME)
    signal = resample_signal(record.p_signal[:, lead_index], record.fs)
    cardiac_beats = [
        (peak, symbol)
        for peak, symbol in zip(annotation.sample, annotation.symbol)
        if symbol in R_PEAK_SYMBOLS
    ]
    beats, rhythm_features, labels = [], [], []
    previous_rr_intervals = []
    for index in range(1, len(cardiac_beats) - 1):
        previous_peak, _ = cardiac_beats[index - 1]
        peak, symbol = cardiac_beats[index]
        next_peak, _ = cardiac_beats[index + 1]
        pre_rr_samples = peak - previous_peak
        post_rr_samples = next_peak - peak
        if pre_rr_samples <= 0 or post_rr_samples <= 0:
            continue
        reference_rr = (
            np.median(previous_rr_intervals[-RHYTHM_HISTORY_SIZE:])
            if previous_rr_intervals
            else pre_rr_samples
        )
        pre_rr_ratio = pre_rr_samples / reference_rr if reference_rr > 0 else 1.0
        previous_rr_intervals.append(pre_rr_samples)
        if symbol not in BEAT_SYMBOLS:
            continue
        resampled_peak = int(round(peak * TARGET_SAMPLE_RATE_HZ / record.fs))
        if resampled_peak < HALF_WINDOW or resampled_peak + HALF_WINDOW >= len(signal):
            continue
        beat = signal[resampled_peak - HALF_WINDOW : resampled_peak + HALF_WINDOW]
        beats.append(normalise_beat(beat).astype(np.float32))
        rhythm_features.append(
            (
                pre_rr_samples / record.fs,
                pre_rr_ratio,
                post_rr_samples / record.fs,
                post_rr_samples / pre_rr_samples,
            )
        )
        labels.append(0 if symbol == "N" else 1)
    return (
        np.asarray(beats, dtype=np.float32).reshape((-1, WINDOW_SIZE, 1)),
        np.asarray(rhythm_features, dtype=np.float32).reshape((-1, 4)),
        np.asarray(labels, dtype=np.int32),
    )


def apply_platt_scaling(raw_probabilities: np.ndarray, coefficient: float, intercept: float) -> np.ndarray:
    clipped = np.clip(raw_probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-(coefficient * logits + intercept)))


def threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "sample_count": int(len(labels)),
        "normal_N_count": int((labels == 0).sum()),
        "pvc_V_count": int((labels == 1).sum()),
        "accuracy": float((tn + tp) / len(labels)) if len(labels) else 0.0,
        "specificity": float(specificity),
        "balanced_accuracy": float((specificity + recall) / 2),
        "pvc_precision": float(precision),
        "pvc_recall": float(recall),
        "pvc_f1": float(2 * precision * recall / (precision + recall) if precision + recall else 0.0),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="External-only INCART evaluation of the frozen PVC model")
    parser.add_argument("--data-dir", type=Path, default=Path("data/incartdb"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("models/pvc_prepost_rr_frozen_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/incart_external_frozen_metrics.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model_path = Path(manifest["model_path"])
    expected_hash = manifest["model_sha256"]
    actual_hash = file_sha256(model_path)
    if actual_hash != expected_hash:
        raise RuntimeError("Frozen model SHA-256 mismatch; external evaluation is blocked")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"INCART data directory not found: {args.data_dir}")

    scaler = manifest["input"]["rhythm_features"]
    if scaler != ["pre_rr_seconds", "pre_rr_ratio", "post_rr_seconds", "post_to_pre_rr_ratio"]:
        raise RuntimeError("Unexpected frozen rhythm feature order")
    metric_report = json.loads(Path(manifest["metrics_path"]).read_text(encoding="utf-8"))
    feature_scaler = metric_report["rhythm_feature_scaler"]
    feature_mean = np.asarray(feature_scaler["mean"], dtype=np.float32)
    feature_std = np.asarray(feature_scaler["standard_deviation"], dtype=np.float32)
    postprocessing = manifest["probability_postprocessing"]
    threshold = float(postprocessing["pvc_threshold"])

    model = tf.keras.models.load_model(model_path)
    all_labels, all_raw_probabilities, record_results = [], [], []
    record_bases = sorted(args.data_dir.glob("*.hea"))
    if not record_bases:
        raise FileNotFoundError(f"No INCART .hea files found in {args.data_dir}")
    for header_path in record_bases:
        waveforms, rhythm_features, labels = extract_record(header_path.with_suffix(""))
        if not len(labels):
            record_results.append({"record": header_path.stem, "status": "no_N_or_V_beats"})
            continue
        inputs = {
            "ecg_beat": waveforms,
            "rr_features": (rhythm_features - feature_mean) / feature_std,
        }
        raw_probabilities = model.predict(inputs, verbose=0)[:, 1]
        calibrated_probabilities = apply_platt_scaling(
            raw_probabilities,
            float(postprocessing["coefficient"]),
            float(postprocessing["intercept"]),
        )
        result = {"record": header_path.stem}
        result.update(threshold_metrics(labels, calibrated_probabilities, threshold))
        if len(np.unique(labels)) == 2:
            result.update(probability_metrics(labels, calibrated_probabilities))
        record_results.append(result)
        all_labels.append(labels)
        all_raw_probabilities.append(raw_probabilities)

    labels = np.concatenate(all_labels)
    raw_probabilities = np.concatenate(all_raw_probabilities)
    probabilities = apply_platt_scaling(
        raw_probabilities,
        float(postprocessing["coefficient"]),
        float(postprocessing["intercept"]),
    )
    pooled_metrics = threshold_metrics(labels, probabilities, threshold)
    pooled_metrics.update(probability_metrics(labels, probabilities))
    output = {
        "evaluation_type": "frozen external test only",
        "model_manifest": str(args.manifest),
        "model_sha256_verified": actual_hash,
        "external_database": "St Petersburg INCART 12-lead Arrhythmia Database",
        "lead": LEAD_NAME,
        "resampling": {"source_hz": 257, "target_hz": TARGET_SAMPLE_RATE_HZ},
        "label_definition": {"0": "normal_N", "1": "PVC_V", "excluded": "all annotation symbols except N and V"},
        "decision_timing": "Each beat is classified after its following R peak arrives",
        "fixed_postprocessing": postprocessing,
        "pooled_metrics": pooled_metrics,
        "records": record_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved frozen external evaluation to {args.output}")


if __name__ == "__main__":
    main()
