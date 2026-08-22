"""Run the fixed Experiment 5 oracle-versus-automatic R-peak comparison.

Only the Gate S2-passed P0/O1 M1/L2 int8 artifact is evaluated.  The classifier,
Platt map, and threshold are immutable; no source, including SVDB or NSRDB,
can enter fitting or selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb

from multisource_ecg import SOURCE_BY_KEY, TARGET_SAMPLE_RATE_HZ, load_pvc_windows, normalise_window, resample_channels, select_leads


MODEL_SHA256 = "edea13aacdd5f6f9f94a3b73092f567f25b4dcade6133da4af7eb42aa2913776"
CALIBRATION_SHA256 = "1525a1988a25021e3398a0cee5bef66263c30d31d24c159cdb730a94dcba59fa"
MANIFEST_SHA256 = "6a091daa2f32fbb45b33772ef8fda7029988741cf57adb18b4e5b3f336f1c6bd"
THRESHOLD = 0.49
TOLERANCE_MS = 150.0
TOLERANCE_SAMPLES = int(round(TOLERANCE_MS * TARGET_SAMPLE_RATE_HZ / 1000.0))
HALF_WINDOW = 150
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_RESAMPLES = 2000
EXPECTED = {"mitdb": 48, "incartdb": 75, "svdb": 78, "nsrdb": 18}
ROLES = {"mitdb": "training_development", "incartdb": "training_development", "svdb": "locked_external_test", "nsrdb": "normal_rhythm_evaluation_only"}


def load_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


experiment3 = load_module("32_experiment3_frozen_svdb.py", "experiment5_experiment3")
legacy_detector = load_module("06_evaluate_streaming_rpeaks.py", "experiment5_detector")
base, core = experiment3.base, experiment3.core


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Experiment 5 artifact: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def records_from_manifest(path: Path) -> list[dict]:
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source"] in EXPECTED:
                if row["role"] != ROLES[row["source"]]:
                    raise ValueError(f"Role mismatch for {row['source']}:{row['record']}")
                records.append(row)
    for source, expected in EXPECTED.items():
        actual = sum(row["source"] == source for row in records)
        if actual != expected:
            raise ValueError(f"Expected {expected} {source} records, found {actual}")
    return sorted(records, key=lambda row: (row["source"], row["record"]))


def rr_features(peaks: np.ndarray, history_peaks: np.ndarray) -> np.ndarray:
    result = []
    for peak in peaks:
        index = int(np.searchsorted(history_peaks, peak, side="left"))
        preceding = history_peaks[:index]
        if len(preceding):
            pre = float(peak - preceding[-1]) / TARGET_SAMPLE_RATE_HZ
            previous = np.diff(preceding[-2:]) / TARGET_SAMPLE_RATE_HZ
            prev = float(previous[-1]) if len(previous) else pre
            history = np.diff(preceding[-9:]) / TARGET_SAMPLE_RATE_HZ
            median = float(np.median(history)) if len(history) else pre
            valid_pre, valid_history = 1.0, float(len(history) >= 8)
        else:
            pre, prev, median, valid_pre, valid_history = 1.0, 1.0, 1.0, 0.0, 0.0
        pre = float(np.clip(pre, 0.2, 3.0))
        result.append([pre, pre / max(median, 0.2), pre / max(prev, 0.2), valid_pre, valid_history])
    return np.asarray(result, dtype=np.float32)


def binary(counts: Counter) -> dict:
    tn, fp, fn, tp = (int(counts[key]) for key in ("tn", "fp", "fn", "tp"))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {"confusion_matrix": [[tn, fp], [fn, tp]], "pvc_precision": precision, "pvc_recall": recall, "pvc_f1": 2 * precision * recall / max(precision + recall, 1e-12), "specificity": specificity, "balanced_accuracy": 0.5 * (recall + specificity)}


def detector_metrics(matches: dict, missed: list[int], unmatched: list[int], reference: list[tuple[int, str]], detected) -> dict:
    tp, fp, fn = len(matches), len(unmatched), len(missed)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    offsets = [(detected[d].peak_sample - reference[r][0]) * 1000.0 / TARGET_SAMPLE_RATE_HZ for r, d in matches.items()]
    absolute = np.abs(np.asarray(offsets, dtype=float))
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, "sensitivity": recall, "ppv": precision, "f1": 2 * precision * recall / max(precision + recall, 1e-12), "absolute_localization_error_ms": {"count": len(absolute), "median": float(np.median(absolute)) if len(absolute) else None, "p95": float(np.quantile(absolute, .95)) if len(absolute) else None, "mean": float(np.mean(absolute)) if len(absolute) else None}}


def evaluate_record(row: dict, data_root: Path, model: Path, scaler) -> tuple[dict, dict]:
    source, record = row["source"], row["record"]
    spec, record_base = SOURCE_BY_KEY[source], data_root / SOURCE_BY_KEY[source].directory / record
    accepted = load_pvc_windows(record_base, spec, lead_count=2)
    annotation = wfdb.rdann(str(record_base), "atr")
    header = wfdb.rdheader(str(record_base))
    all_annotation_peaks = np.unique(np.rint(annotation.sample * TARGET_SAMPLE_RATE_HZ / header.fs).astype(np.int64))
    classifier_metrics_available = bool(len(accepted["labels"]))
    oracle_rr = rr_features(accepted["canonical_r_peak_samples"], all_annotation_peaks)
    oracle_raw = base.tflite_predict(model, accepted["waveforms"], oracle_rr).astype(np.float32) if classifier_metrics_available else np.empty(0, dtype=np.float32)
    oracle_probability = core.apply_scaler(scaler, oracle_raw).astype(np.float32) if len(oracle_raw) else oracle_raw

    raw_record = wfdb.rdrecord(str(record_base), channels=select_leads(list(header.sig_name), spec)[1]["channel_indices"])
    signal = resample_channels(raw_record.p_signal, raw_record.fs)
    reference = [(int(round(sample * TARGET_SAMPLE_RATE_HZ / raw_record.fs)), symbol) for sample, symbol in zip(annotation.sample, annotation.symbol) if symbol in legacy_detector.R_PEAK_SYMBOLS]
    detected = legacy_detector.causal_pan_tompkins(signal[:, 0], TARGET_SAMPLE_RATE_HZ)
    matches, missed, unmatched = legacy_detector.match_peaks(reference, detected, TOLERANCE_SAMPLES)
    auto_indices = [index for index, peak in enumerate(detected) if peak.peak_sample >= HALF_WINDOW and peak.peak_sample + HALF_WINDOW < len(signal)]
    auto_peaks = np.asarray([detected[index].peak_sample for index in auto_indices], dtype=np.int64)
    auto_windows = np.asarray([normalise_window(signal[peak - HALF_WINDOW:peak + HALF_WINDOW]) for peak in auto_peaks], dtype=np.float32)
    auto_rr = rr_features(auto_peaks, np.asarray([peak.peak_sample for peak in detected], dtype=np.int64))
    auto_raw = base.tflite_predict(model, auto_windows, auto_rr).astype(np.float32) if len(auto_peaks) else np.empty(0, dtype=np.float32)
    auto_probability = core.apply_scaler(scaler, auto_raw).astype(np.float32) if len(auto_raw) else auto_raw
    auto_by_detection = {index: (auto_probability[position], auto_rr[position]) for position, index in enumerate(auto_indices)}

    oracle_counts, automatic_counts = Counter(), Counter()
    labels, auto_aligned_probability = [], []
    error = Counter()
    reference_lookup = {peak: label for peak, label in zip(accepted["canonical_r_peak_samples"], accepted["labels"])}
    oracle_decision = {peak: int(probability >= THRESHOLD) for peak, probability in zip(accepted["canonical_r_peak_samples"], oracle_probability)}
    for ref_index, (ref_peak, symbol) in enumerate(reference):
        if symbol not in {"N", "V"} or ref_peak not in reference_lookup:
            continue
        label, oracle_prediction = int(reference_lookup[ref_peak]), oracle_decision[ref_peak]
        oracle_counts[("tp" if label and oracle_prediction else "fn" if label else "fp" if oracle_prediction else "tn")] += 1
        detection_index = matches.get(ref_index)
        if detection_index is None or detection_index not in auto_by_detection:
            automatic_counts["fn" if label else "tn"] += 1
            if label:
                error["missed_r_peak"] += 1
            continue
        probability, causal_rr = auto_by_detection[detection_index]
        prediction = int(probability >= THRESHOLD)
        automatic_counts[("tp" if label and prediction else "fn" if label else "fp" if prediction else "tn")] += 1
        labels.append(label); auto_aligned_probability.append(float(probability))
        if label and not prediction:
            # Counterfactual uses the same automatic RR feature with only the window centre restored.
            reference_window = normalise_window(signal[ref_peak - HALF_WINDOW:ref_peak + HALF_WINDOW])[None, ...]
            aligned = core.apply_scaler(scaler, base.tflite_predict(model, reference_window, causal_rr[None, ...]))[0]
            error["mislocalized_peak_window"] += int(aligned >= THRESHOLD)
            error["classifier_or_causal_rr_error"] += int(aligned < THRESHOLD)
    automatic_unmatched_positive_alarms = 0
    for detection_index in unmatched:
        if detection_index in auto_by_detection and auto_by_detection[detection_index][0] >= THRESHOLD:
            if classifier_metrics_available:
                automatic_counts["fp"] += 1
            else:
                automatic_unmatched_positive_alarms += 1
            error["unmatched_peak_false_alarm"] += 1

    per_record = {"record_key": f"{source}:{record}", "source": source, "classifier_metrics_available": classifier_metrics_available, "classifier_metrics_unavailable_reason": None if classifier_metrics_available else "no_eligible_n_v_window_after_fixed_boundary_exclusion", "oracle": binary(oracle_counts) if classifier_metrics_available else None, "automatic": binary(automatic_counts) if classifier_metrics_available else None, "r_peak_detection": detector_metrics(matches, missed, unmatched, reference, detected), "error_decomposition": dict(error), "matched_labeled_beats_for_probability": len(labels), "automatic_unmatched_positive_alarms_without_reference_classification": automatic_unmatched_positive_alarms}
    archive = {f"{source}__{record}__labels": accepted["labels"].astype(np.int8), f"{source}__{record}__oracle_probability": oracle_probability, f"{source}__{record}__automatic_probability_aligned": np.asarray(auto_aligned_probability, dtype=np.float32)}
    return per_record, archive


def aggregate(rows: list[dict]) -> dict:
    detector, oracle, automatic, errors = Counter(), Counter(), Counter(), Counter()
    classification_rows = 0
    unavailable_unmatched_positive_alarms = 0
    for row in rows:
        if row["classifier_metrics_available"]:
            classification_rows += 1
            for name, target in (("oracle", oracle), ("automatic", automatic)):
                matrix = row[name]["confusion_matrix"]
                target.update({"tn": matrix[0][0], "fp": matrix[0][1], "fn": matrix[1][0], "tp": matrix[1][1]})
        else:
            unavailable_unmatched_positive_alarms += row["automatic_unmatched_positive_alarms_without_reference_classification"]
        r = row["r_peak_detection"]; detector.update({"tp": r["true_positive"], "fp": r["false_positive"], "fn": r["false_negative"]})
        errors.update(row["error_decomposition"])
    precision, recall = detector["tp"] / max(detector["tp"] + detector["fp"], 1), detector["tp"] / max(detector["tp"] + detector["fn"], 1)
    oracle_summary = binary(oracle) if classification_rows else None
    automatic_summary = binary(automatic) if classification_rows else None
    return {"record_count": len(rows), "classifier_metric_record_count": classification_rows, "classifier_metric_unavailable_record_count": len(rows) - classification_rows, "automatic_unmatched_positive_alarms_in_unavailable_classifier_records": unavailable_unmatched_positive_alarms, "r_peak_detection": {"true_positive": detector["tp"], "false_positive": detector["fp"], "false_negative": detector["fn"], "sensitivity": recall, "ppv": precision, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}, "oracle_classification": oracle_summary, "automatic_classification": automatic_summary, "absolute_degradation_automatic_minus_oracle": {key: automatic_summary[key] - oracle_summary[key] for key in ("pvc_precision", "pvc_recall", "pvc_f1", "specificity", "balanced_accuracy")} if classification_rows else None, "pvc_miss_error_decomposition": dict(errors)}


def bootstrap(rows: list[dict]) -> dict:
    classifier_rows = [row for row in rows if row["classifier_metrics_available"]]
    if not classifier_rows:
        return {"classifier_metric_record_count": 0, "bootstrap_resamples": 0, "seed": BOOTSTRAP_SEED, "reason": "no_records_with_eligible_n_v_reference_windows"}
    rng, values = np.random.default_rng(BOOTSTRAP_SEED), {"oracle_f1": [], "automatic_f1": [], "oracle_recall": [], "automatic_recall": []}
    for _ in range(BOOTSTRAP_RESAMPLES):
        picked = [classifier_rows[int(i)] for i in rng.integers(0, len(classifier_rows), len(classifier_rows))]
        summary = aggregate(picked)
        for metric, path in (("oracle_f1", ("oracle_classification", "pvc_f1")), ("automatic_f1", ("automatic_classification", "pvc_f1")), ("oracle_recall", ("oracle_classification", "pvc_recall")), ("automatic_recall", ("automatic_classification", "pvc_recall"))):
            values[metric].append(summary[path[0]][path[1]])
    return {"classifier_metric_record_count": len(classifier_rows), **{key: {"bootstrap_resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "ci_95": [float(np.quantile(value, .025)), float(np.quantile(value, .975))]} for key, value in values.items()}}


def write_csv(path: Path, rows: list[dict]) -> None:
    flat = []
    for row in rows:
        flat.append({"record_key": row["record_key"], "source": row["source"], "classifier_metrics_available": row["classifier_metrics_available"], "classifier_metrics_unavailable_reason": row["classifier_metrics_unavailable_reason"], "automatic_unmatched_positive_alarms_without_reference_classification": row["automatic_unmatched_positive_alarms_without_reference_classification"], **{f"detector_{k}": v for k, v in row["r_peak_detection"].items() if not isinstance(v, dict)}, **{f"oracle_{k}": v for k, v in (row["oracle"] or {}).items() if k != "confusion_matrix"}, **{f"automatic_{k}": v for k, v in (row["automatic"] or {}).items() if k != "confusion_matrix"}, **row["error_decomposition"]})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in flat for key in row}))
        writer.writeheader(); writer.writerows(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/gate_s2_int8_tf215_run1/model_int8.tflite"))
    parser.add_argument("--calibration", type=Path, default=Path("models/gate_s2_int8_tf215_run1/calibration.json"))
    parser.add_argument("--freeze-manifest", type=Path, default=Path("results/gate_s2_int8_tf215_run1/manifest.json"))
    parser.add_argument("--record-manifest", type=Path, default=Path("results/experiment0/record_manifest.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--result-dir", type=Path, default=Path("results/experiment5"))
    parser.add_argument("--allow-frozen-rpeak-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.allow_frozen_rpeak_evaluation:
        parser.error("Experiment 5 requires --allow-frozen-rpeak-evaluation")
    if args.result_dir.exists():
        raise FileExistsError(f"Experiment 5 output already exists: {args.result_dir}")
    if (sha256(args.model), sha256(args.calibration), sha256(args.freeze_manifest)) != (MODEL_SHA256, CALIBRATION_SHA256, MANIFEST_SHA256):
        raise ValueError("Frozen Gate S2 artifact hash mismatch")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8")); manifest = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed" or float(calibration.get("threshold", -1)) != THRESHOLD:
        raise ValueError("Gate S2 status or frozen threshold mismatch")
    args.result_dir.mkdir(parents=True)
    protocol = {"experiment": "experiment_5_oracle_versus_automatic_rpeak", "status": "registered_execution", "started_at_utc": datetime.now(timezone.utc).isoformat(), "candidate": "P0/O1", "model_sha256": MODEL_SHA256, "calibration_sha256": CALIBRATION_SHA256, "freeze_manifest_sha256": MANIFEST_SHA256, "threshold": THRESHOLD, "sources": EXPECTED, "r_peak_detector": "existing causal Pan-Tompkins-style implementation from src/06_evaluate_streaming_rpeaks.py", "r_peak_match_tolerance_ms": TOLERANCE_MS, "automatic_detector_channel": "first frozen two-channel lead", "prohibited_operations": ["retraining", "detector tuning", "calibration fitting", "threshold selection", "model selection"], "miss_decomposition": "missed_r_peak: unmatched V reference; mislocalized_peak_window: automatic-centre inference missed but reference-centre inference with identical automatic RR passed; classifier_or_causal_rr_error: both inference centres missed", "write_once": True}
    write_json(args.result_dir / "protocol.json", protocol)
    scaler, rows, arrays = experiment3.frozen_scaler(calibration), [], {}
    for index, row in enumerate(records_from_manifest(args.record_manifest), start=1):
        metrics, archive = evaluate_record(row, args.data_root, args.model, scaler)
        rows.append(metrics); arrays.update(archive)
        print(f"[{index}/219] {metrics['record_key']}", flush=True)
    by_source = {source: aggregate([row for row in rows if row["source"] == source]) for source in EXPECTED}
    summary = {"experiment": protocol["experiment"], "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "external_data_accessed": True, "frozen_configuration_verified": True, "candidate": "P0/O1", "model_sha256": MODEL_SHA256, "calibration_sha256": CALIBRATION_SHA256, "threshold": THRESHOLD, "all_sources": aggregate(rows), "by_source": by_source, "record_level_bootstrap_ci_95": bootstrap(rows), "artifacts": {"protocol": str(args.result_dir / "protocol.json"), "per_record_metrics": str(args.result_dir / "per_record_metrics.csv"), "per_record_predictions": str(args.result_dir / "per_record_predictions.npz")}, "source_sha256": sha256(Path(__file__)), "dependencies": {"python": sys.version, "platform": platform.platform()}, "post_evaluation_rule": "No model, detector, calibration, threshold, quantization, preprocessing, lead profile, or seed change may be selected from these results."}
    write_csv(args.result_dir / "per_record_metrics.csv", rows)
    np.savez_compressed(args.result_dir / "per_record_predictions.npz", **arrays)
    write_json(args.result_dir / "summary.json", summary)
    print(json.dumps({"status": "complete", "summary": str(args.result_dir / "summary.json"), "records": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
