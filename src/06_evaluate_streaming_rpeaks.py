"""Compare oracle and causal Pan-Tompkins R peaks for a frozen PVC model.

The model was trained on 360 Hz, 300-sample waveform windows.  This evaluator
therefore resamples external records to 360 Hz before running a causal
detector, and keeps all model calibration and thresholds fixed.  It reports
desktop runtime separately from the algorithmic delay caused by acquiring the
post-R portion of the model window.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import tensorflow as tf
import wfdb
from scipy.signal import butter, resample_poly, sosfilt
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


WINDOW_SIZE = 300
HALF_WINDOW = WINDOW_SIZE // 2
TARGET_SAMPLE_RATE_HZ = 360
BEAT_SYMBOLS = {"N", "V"}
R_PEAK_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j",
    "n", "E", "/", "f", "Q", "?", "P", "U",
}


@dataclass(frozen=True)
class DetectedPeak:
    peak_sample: int
    emitted_sample: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalise_beat(beat: np.ndarray) -> np.ndarray:
    standard_deviation = beat.std()
    return (beat - beat.mean()) / (standard_deviation if standard_deviation > 1e-8 else 1.0)


def resample_signal(signal: np.ndarray, source_hz: float) -> np.ndarray:
    if source_hz == TARGET_SAMPLE_RATE_HZ:
        return signal.astype(np.float32, copy=False)
    ratio = Fraction(TARGET_SAMPLE_RATE_HZ / source_hz).limit_denominator(10000)
    return resample_poly(signal, ratio.numerator, ratio.denominator).astype(np.float32)


def select_signal(record: wfdb.Record, lead_name: str | None) -> tuple[np.ndarray, str]:
    if lead_name is not None:
        if lead_name not in record.sig_name:
            raise ValueError(f"Record has no requested lead {lead_name!r}: {record.sig_name}")
        index = record.sig_name.index(lead_name)
    else:
        index = 0
    return record.p_signal[:, index], record.sig_name[index]


def load_record(record_base: Path, lead_name: str | None) -> tuple[np.ndarray, list[tuple[int, str]], str, float]:
    record = wfdb.rdrecord(str(record_base))
    annotation = wfdb.rdann(str(record_base), "atr")
    source_signal, selected_lead = select_signal(record, lead_name)
    signal = resample_signal(source_signal, record.fs)
    scale = TARGET_SAMPLE_RATE_HZ / record.fs
    reference_peaks = [
        (int(round(sample * scale)), symbol)
        for sample, symbol in zip(annotation.sample, annotation.symbol)
        if symbol in R_PEAK_SYMBOLS
    ]
    return signal, reference_peaks, selected_lead, record.fs


def causal_pan_tompkins(signal: np.ndarray, sample_rate_hz: int) -> list[DetectedPeak]:
    """Return R-peak candidates using only samples available at emission time."""
    sos = butter(2, (5.0, 15.0), btype="bandpass", fs=sample_rate_hz, output="sos")
    filtered = sosfilt(sos, signal.astype(np.float64, copy=False))
    derivative = np.empty_like(filtered)
    derivative[0] = 0.0
    derivative[1:] = np.diff(filtered)
    squared = np.square(derivative)
    integration_width = max(1, int(round(0.150 * sample_rate_hz)))
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    integrated = np.empty_like(squared)
    for index in range(len(squared)):
        start = max(0, index + 1 - integration_width)
        integrated[index] = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)

    initial_end = min(len(integrated), max(integration_width * 2, 2 * sample_rate_hz))
    initial = integrated[:initial_end]
    signal_level = float(np.percentile(initial, 95)) if len(initial) else 0.0
    noise_level = float(np.median(initial)) if len(initial) else 0.0
    threshold = noise_level + 0.25 * (signal_level - noise_level)
    refractory_samples = int(round(0.200 * sample_rate_hz))
    peaks: list[DetectedPeak] = []
    last_peak = -refractory_samples

    # At ``index`` the candidate at index - 1 is known to be a local maximum;
    # no future sample beyond the emitted sample has influenced this decision.
    for index in range(max(1, initial_end), len(integrated)):
        candidate = index - 1
        value = integrated[candidate]
        if not (value >= integrated[candidate - 1] and value > integrated[index]):
            continue
        if candidate - last_peak < refractory_samples:
            noise_level = 0.125 * value + 0.875 * noise_level
            threshold = noise_level + 0.25 * (signal_level - noise_level)
            continue
        if value < threshold:
            noise_level = 0.125 * value + 0.875 * noise_level
            threshold = noise_level + 0.25 * (signal_level - noise_level)
            continue
        search_start = max(0, candidate - integration_width)
        search_stop = candidate + 1
        r_peak = search_start + int(np.argmax(np.abs(filtered[search_start:search_stop])))
        if r_peak - last_peak >= refractory_samples:
            peaks.append(DetectedPeak(r_peak, index))
            last_peak = r_peak
            signal_level = 0.125 * value + 0.875 * signal_level
        else:
            noise_level = 0.125 * value + 0.875 * noise_level
        threshold = noise_level + 0.25 * (signal_level - noise_level)
    return peaks


def match_peaks(
    reference_peaks: list[tuple[int, str]],
    detected_peaks: list[DetectedPeak],
    tolerance_samples: int,
) -> tuple[dict[int, int], list[int], list[int]]:
    """Greedily match chronological peaks one-to-one within a fixed tolerance."""
    matches: dict[int, int] = {}
    missed_reference: list[int] = []
    unmatched_detected: list[int] = []
    detected_index = 0
    for reference_index, (reference_sample, _) in enumerate(reference_peaks):
        while (
            detected_index < len(detected_peaks)
            and detected_peaks[detected_index].peak_sample < reference_sample - tolerance_samples
        ):
            unmatched_detected.append(detected_index)
            detected_index += 1
        if detected_index >= len(detected_peaks):
            missed_reference.append(reference_index)
            continue
        candidate_index = detected_index
        candidate_distance = abs(detected_peaks[candidate_index].peak_sample - reference_sample)
        next_index = candidate_index + 1
        if (
            next_index < len(detected_peaks)
            and abs(detected_peaks[next_index].peak_sample - reference_sample) <= tolerance_samples
            and abs(detected_peaks[next_index].peak_sample - reference_sample) < candidate_distance
        ):
            unmatched_detected.append(candidate_index)
            candidate_index = next_index
            candidate_distance = abs(detected_peaks[candidate_index].peak_sample - reference_sample)
        if candidate_distance <= tolerance_samples:
            matches[reference_index] = candidate_index
            detected_index = candidate_index + 1
        else:
            missed_reference.append(reference_index)
    unmatched_detected.extend(range(detected_index, len(detected_peaks)))
    return matches, missed_reference, unmatched_detected


def apply_platt_scaling(raw_probabilities: np.ndarray, coefficient: float, intercept: float) -> np.ndarray:
    clipped = np.clip(raw_probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-(coefficient * logits + intercept)))


def predict_windows(model: tf.keras.Model, windows: list[np.ndarray], batch_size: int) -> tuple[np.ndarray, float]:
    if not windows:
        return np.empty(0, dtype=np.float32), 0.0
    model_input = np.asarray(windows, dtype=np.float32).reshape((-1, WINDOW_SIZE, 1))
    started = time.perf_counter()
    raw_probabilities = model.predict(model_input, batch_size=batch_size, verbose=0)[:, 1]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return raw_probabilities.astype(np.float32), elapsed_ms / len(model_input)


def binary_metrics(counts: Counter) -> dict:
    true_negative = counts["tn"]
    false_positive = counts["fp"]
    false_negative = counts["fn"]
    true_positive = counts["tp"]
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    return {
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
        "pvc_precision": precision,
        "pvc_recall": recall,
        "pvc_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": specificity,
        "balanced_accuracy": (specificity + recall) / 2,
    }


def summarise(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(array.max()),
    }


def evaluate_record(
    record_base: Path,
    lead_name: str | None,
    model: tf.keras.Model,
    threshold: float,
    coefficient: float,
    intercept: float,
    tolerance_ms: float,
    batch_size: int,
) -> dict:
    signal, reference_peaks, selected_lead, source_hz = load_record(record_base, lead_name)
    started = time.perf_counter()
    detected_peaks = causal_pan_tompkins(signal, TARGET_SAMPLE_RATE_HZ)
    detector_elapsed_ms = (time.perf_counter() - started) * 1000.0
    tolerance_samples = int(round(tolerance_ms * TARGET_SAMPLE_RATE_HZ / 1000.0))
    matches, missed_reference, unmatched_detected = match_peaks(
        reference_peaks,
        detected_peaks,
        tolerance_samples,
    )

    oracle_reference_indices = [
        index for index, (peak, symbol) in enumerate(reference_peaks)
        if symbol in BEAT_SYMBOLS and peak >= HALF_WINDOW and peak + HALF_WINDOW < len(signal)
    ]
    oracle_windows = [normalise_beat(signal[reference_peaks[index][0] - HALF_WINDOW:reference_peaks[index][0] + HALF_WINDOW]) for index in oracle_reference_indices]
    oracle_raw, oracle_inference_ms = predict_windows(model, oracle_windows, batch_size)
    oracle_probabilities = apply_platt_scaling(oracle_raw, coefficient, intercept)
    oracle_labels = np.asarray([0 if reference_peaks[index][1] == "N" else 1 for index in oracle_reference_indices], dtype=int)

    auto_detected_indices = [
        index for index, peak in enumerate(detected_peaks)
        if peak.peak_sample >= HALF_WINDOW and peak.peak_sample + HALF_WINDOW < len(signal)
    ]
    auto_windows = [normalise_beat(signal[detected_peaks[index].peak_sample - HALF_WINDOW:detected_peaks[index].peak_sample + HALF_WINDOW]) for index in auto_detected_indices]
    auto_raw, auto_inference_ms = predict_windows(model, auto_windows, batch_size)
    auto_probabilities = apply_platt_scaling(auto_raw, coefficient, intercept)
    auto_prediction_by_detection = {
        detected_index: int(probability >= threshold)
        for detected_index, probability in zip(auto_detected_indices, auto_probabilities)
    }

    counts = Counter()
    decision_latencies_ms: list[float] = []
    detector_latencies_ms: list[float] = []
    offsets_ms: list[float] = []
    for reference_index, (reference_sample, symbol) in enumerate(reference_peaks):
        detected_index = matches.get(reference_index)
        if detected_index is None:
            if symbol == "V":
                counts["fn"] += 1
            elif symbol == "N":
                counts["unclassified_normal_reference"] += 1
            continue
        detected_peak = detected_peaks[detected_index]
        offset_ms = (detected_peak.peak_sample - reference_sample) * 1000.0 / TARGET_SAMPLE_RATE_HZ
        offsets_ms.append(offset_ms)
        detector_latency_ms = (detected_peak.emitted_sample - reference_sample) * 1000.0 / TARGET_SAMPLE_RATE_HZ
        detector_latencies_ms.append(detector_latency_ms)
        decision_sample = max(detected_peak.emitted_sample, detected_peak.peak_sample + HALF_WINDOW)
        decision_latencies_ms.append((decision_sample - reference_sample) * 1000.0 / TARGET_SAMPLE_RATE_HZ + auto_inference_ms)
        if symbol not in BEAT_SYMBOLS or detected_index not in auto_prediction_by_detection:
            continue
        prediction = auto_prediction_by_detection[detected_index]
        label = 0 if symbol == "N" else 1
        if label == 1 and prediction == 1:
            counts["tp"] += 1
        elif label == 1:
            counts["fn"] += 1
        elif prediction == 1:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    for detected_index in unmatched_detected:
        if auto_prediction_by_detection.get(detected_index) == 1:
            counts["fp"] += 1
            counts["unmatched_pvc_alarms"] += 1

    detector_true_positive = len(matches)
    detector_false_negative = len(missed_reference)
    detector_false_positive = len(unmatched_detected)
    detector_precision = detector_true_positive / (detector_true_positive + detector_false_positive) if detector_true_positive + detector_false_positive else 0.0
    detector_recall = detector_true_positive / (detector_true_positive + detector_false_negative) if detector_true_positive + detector_false_negative else 0.0
    oracle_metrics = binary_metrics(Counter())
    if len(oracle_labels):
        oracle_predictions = (oracle_probabilities >= threshold).astype(int)
        oracle_counts = Counter()
        oracle_counts["tn"] = int(np.sum((oracle_labels == 0) & (oracle_predictions == 0)))
        oracle_counts["fp"] = int(np.sum((oracle_labels == 0) & (oracle_predictions == 1)))
        oracle_counts["fn"] = int(np.sum((oracle_labels == 1) & (oracle_predictions == 0)))
        oracle_counts["tp"] = int(np.sum((oracle_labels == 1) & (oracle_predictions == 1)))
        oracle_metrics = binary_metrics(oracle_counts)
        if len(np.unique(oracle_labels)) == 2:
            oracle_metrics.update({
                "auroc": float(roc_auc_score(oracle_labels, oracle_probabilities)),
                "auprc": float(average_precision_score(oracle_labels, oracle_probabilities)),
                "brier_score": float(brier_score_loss(oracle_labels, oracle_probabilities)),
            })
    return {
        "record": record_base.name,
        "source_sample_rate_hz": source_hz,
        "selected_lead": selected_lead,
        "signal_samples_at_360_hz": int(len(signal)),
        "reference_r_peaks": int(len(reference_peaks)),
        "detected_r_peaks": int(len(detected_peaks)),
        "r_peak_detection": {
            "tolerance_ms": tolerance_ms,
            "true_positive": detector_true_positive,
            "false_positive": detector_false_positive,
            "false_negative": detector_false_negative,
            "sensitivity": detector_recall,
            "ppv": detector_precision,
            "f1": 2 * detector_precision * detector_recall / (detector_precision + detector_recall) if detector_precision + detector_recall else 0.0,
            "signed_peak_offset_ms": summarise(offsets_ms),
            "absolute_peak_error_ms": summarise([abs(value) for value in offsets_ms]),
        },
        "oracle_r_classification": oracle_metrics,
        "automatic_r_classification": {**binary_metrics(counts), **{key: int(value) for key, value in counts.items()}},
        "latency_ms": {
            "detector_output_relative_to_reference_r": summarise(detector_latencies_ms),
            "algorithmic_decision_plus_desktop_batch_inference": summarise(decision_latencies_ms),
            "window_post_r_observation_wait": HALF_WINDOW * 1000.0 / TARGET_SAMPLE_RATE_HZ,
            "detector_desktop_cpu_ms_total": detector_elapsed_ms,
            "detector_desktop_cpu_ms_per_input_sample": detector_elapsed_ms / len(signal),
            "oracle_desktop_batch_inference_ms_per_window": oracle_inference_ms,
            "automatic_desktop_batch_inference_ms_per_window": auto_inference_ms,
        },
    }


def aggregate_records(records: list[dict]) -> dict:
    detector = Counter()
    automatic = Counter()
    oracle = Counter()
    offsets, detector_latencies, decision_latencies = [], [], []
    for record in records:
        detector_metrics = record["r_peak_detection"]
        detector["tp"] += detector_metrics["true_positive"]
        detector["fp"] += detector_metrics["false_positive"]
        detector["fn"] += detector_metrics["false_negative"]
        automatic_metrics = record["automatic_r_classification"]
        oracle_metrics = record["oracle_r_classification"]
        for key, source in (("tn", automatic_metrics), ("fp", automatic_metrics), ("fn", automatic_metrics), ("tp", automatic_metrics)):
            automatic[key] += int(source.get(key, 0))
            oracle[key] += int(oracle_metrics["confusion_matrix"][0][0] if key == "tn" else oracle_metrics["confusion_matrix"][0][1] if key == "fp" else oracle_metrics["confusion_matrix"][1][0] if key == "fn" else oracle_metrics["confusion_matrix"][1][1])
        offsets.extend(record["r_peak_detection"]["signed_peak_offset_ms"].get("mean", []) if False else [])
        detector_latencies.append(record["latency_ms"]["detector_output_relative_to_reference_r"].get("median", np.nan))
        decision_latencies.append(record["latency_ms"]["algorithmic_decision_plus_desktop_batch_inference"].get("median", np.nan))
    detector_precision = detector["tp"] / (detector["tp"] + detector["fp"]) if detector["tp"] + detector["fp"] else 0.0
    detector_recall = detector["tp"] / (detector["tp"] + detector["fn"]) if detector["tp"] + detector["fn"] else 0.0
    return {
        "record_count": len(records),
        "r_peak_detection": {
            "true_positive": detector["tp"], "false_positive": detector["fp"], "false_negative": detector["fn"],
            "sensitivity": detector_recall, "ppv": detector_precision,
            "f1": 2 * detector_precision * detector_recall / (detector_precision + detector_recall) if detector_precision + detector_recall else 0.0,
        },
        "oracle_r_classification": binary_metrics(oracle),
        "automatic_r_classification": binary_metrics(automatic),
        "per_record_median_detector_latency_ms": summarise([value for value in detector_latencies if np.isfinite(value)]),
        "per_record_median_decision_latency_ms": summarise([value for value in decision_latencies if np.isfinite(value)]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen PVC evaluation with oracle and causal automatic R peaks")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-role", choices=("development_overlap", "external_test_only"), required=True)
    parser.add_argument("--lead-name", default=None, help="Exact lead name; defaults to the first signal channel")
    parser.add_argument("--model", type=Path, default=Path("models/tiny_1dcnn_pvc_calibrated.keras"))
    parser.add_argument("--metrics", type=Path, default=Path("models/pvc_calibrated_metrics.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-ms", type=float, default=75.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit-records", type=int, default=None)
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if args.tolerance_ms <= 0:
        parser.error("--tolerance-ms must be positive")
    metric_report = json.loads(args.metrics.read_text(encoding="utf-8"))
    deployment = metric_report["final_deployment_model"]
    if metric_report.get("input_features", "waveform only") != "waveform only":
        raise RuntimeError("This evaluator only supports the causal waveform-only deployment model")
    coefficient = float(deployment["calibration_coefficient"])
    intercept = float(deployment["calibration_intercept"])
    threshold = float(deployment["pvc_threshold"])
    model_hash = file_sha256(args.model)
    model = tf.keras.models.load_model(args.model)
    record_bases = sorted(path.with_suffix("") for path in args.data_dir.glob("*.hea"))
    if args.limit_records is not None:
        record_bases = record_bases[:args.limit_records]
    if not record_bases:
        raise FileNotFoundError(f"No WFDB .hea records found in {args.data_dir}")
    records = [
        evaluate_record(
            record_base, args.lead_name, model, threshold, coefficient, intercept,
            args.tolerance_ms, args.batch_size,
        )
        for record_base in record_bases
    ]
    result = {
        "evaluation_type": "frozen waveform-only PVC model with oracle-versus-causal-automatic R peaks",
        "dataset": {"name": args.dataset_name, "role": args.dataset_role, "path": str(args.data_dir)},
        "model": {
            "path": str(args.model), "sha256": model_hash,
            "calibration_source": str(args.metrics), "calibration_coefficient": coefficient,
            "calibration_intercept": intercept, "pvc_threshold": threshold,
        },
        "causal_protocol": {
            "detector": "causal Pan-Tompkins-style bandpass, derivative, square, moving integration, adaptive threshold",
            "model_sampling_rate_hz": TARGET_SAMPLE_RATE_HZ,
            "window_samples": WINDOW_SIZE,
            "post_r_observation_wait_ms": HALF_WINDOW * 1000.0 / TARGET_SAMPLE_RATE_HZ,
            "r_peak_match_tolerance_ms": args.tolerance_ms,
            "desktop_runtime_disclaimer": "Batch inference timings are desktop benchmarks, not MCU latency measurements.",
            "external_test_policy": "No training, threshold selection, calibration fitting, or detector parameter tuning on external records.",
        },
        "summary": aggregate_records(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved evaluation to {args.output}")


if __name__ == "__main__":
    main()
