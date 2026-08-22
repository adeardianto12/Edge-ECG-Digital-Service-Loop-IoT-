"""Execute the one-time frozen Experiment 3 SVDB evaluation.

This program deliberately has no fitting, calibration, threshold-selection, or
model-selection path.  It evaluates only the Gate S2-passed full-int8 P0/O1,
M1/L2 deployment artifact on the locked SVDB records under the fixed oracle
annotation-window preprocessing contract.  Automatic R-peak evaluation is
reserved for Experiment 5.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb
from sklearn.linear_model import LogisticRegression

from multisource_ecg import SOURCE_BY_KEY, TARGET_SAMPLE_RATE_HZ, load_pvc_windows


FROZEN_CANDIDATE = "P0/O1"
FROZEN_SOURCE_CONDITION = "M1: MIT-BIH + INCART"
FROZEN_LEAD_PROFILE = "L2"
FROZEN_PARAMETER_COUNT = 3506
FROZEN_THRESHOLD = 0.49
EXPECTED_MODEL_SHA256 = "edea13aacdd5f6f9f94a3b73092f567f25b4dcade6133da4af7eb42aa2913776"
EXPECTED_CALIBRATION_SHA256 = "1525a1988a25021e3398a0cee5bef66263c30d31d24c159cdb730a94dcba59fa"
EXPECTED_MANIFEST_SHA256 = "6a091daa2f32fbb45b33772ef8fda7029988741cf57adb18b4e5b3f336f1c6bd"
EXPECTED_RECORD_COUNT = 78
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_RESAMPLES = 2000


def load_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate_s2 = load_module("31_gate_s2_int8_calibration.py", "experiment3_gate_s2")
base = gate_s2.base
core = gate_s2.core


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Experiment 3 artifact: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def frozen_scaler(calibration: dict) -> LogisticRegression:
    scaler = LogisticRegression()
    scaler.classes_ = np.asarray([0, 1], dtype=np.int32)
    scaler.coef_ = np.asarray([[float(calibration["coefficient"])]], dtype=np.float64)
    scaler.intercept_ = np.asarray([float(calibration["intercept"])], dtype=np.float64)
    scaler.n_features_in_ = 1
    return scaler


def load_svdb_records(manifest_path: Path) -> list[dict]:
    records: list[dict] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source"] != "svdb":
                continue
            if row["role"] != "locked_external_test":
                raise ValueError("SVDB record does not have locked_external_test role")
            records.append(row)
    if len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError(f"Expected {EXPECTED_RECORD_COUNT} SVDB records, found {len(records)}")
    names = [row["record"] for row in records]
    if len(names) != len(set(names)):
        raise ValueError("SVDB manifest contains duplicate record identifiers")
    return sorted(records, key=lambda row: row["record"])


def causal_rr_features(record_base: Path, lead_count: int) -> tuple[dict, np.ndarray]:
    """Reproduce the frozen past-only RR feature construction for one record."""
    spec = SOURCE_BY_KEY["svdb"]
    header = wfdb.rdheader(str(record_base))
    annotation = wfdb.rdann(str(record_base), "atr")
    all_peaks = np.unique(np.rint(annotation.sample * TARGET_SAMPLE_RATE_HZ / header.fs).astype(np.int64))
    accepted = load_pvc_windows(record_base, spec, lead_count=lead_count)
    output = []
    for peak in accepted["canonical_r_peak_samples"]:
        index = int(np.searchsorted(all_peaks, peak, side="left"))
        preceding = all_peaks[:index]
        if len(preceding):
            pre_rr = float(peak - preceding[-1]) / TARGET_SAMPLE_RATE_HZ
            previous = np.diff(preceding[-2:]) / TARGET_SAMPLE_RATE_HZ
            prev_rr = float(previous[-1]) if len(previous) else pre_rr
            history = np.diff(preceding[-9:]) / TARGET_SAMPLE_RATE_HZ
            median = float(np.median(history)) if len(history) else pre_rr
            valid_pre = 1.0
            valid_history = float(len(history) >= 8)
        else:
            pre_rr, prev_rr, median, valid_pre, valid_history = 1.0, 1.0, 1.0, 0.0, 0.0
        pre_rr = float(np.clip(pre_rr, 0.2, 3.0))
        output.append([pre_rr, pre_rr / max(median, 0.2), pre_rr / max(prev_rr, 0.2), valid_pre, valid_history])
    rr = np.asarray(output, dtype=np.float32)
    if len(rr) != len(accepted["labels"]):
        raise ValueError(f"RR/window count mismatch for {record_base.name}")
    return accepted, rr


def record_macro_extended(per_record: dict[str, dict]) -> dict:
    items = list(per_record.values())
    support = [item for item in items if item["has_pvc_reference"]]
    both = [item for item in items if item["auroc"] is not None]

    def mean(items: list[dict], field: str):
        values = [item[field] for item in items if item[field] is not None]
        return float(np.mean(values)) if values else None

    balanced = [0.5 * (item["pvc_recall"] + item["specificity"]) for item in support]
    return {
        "support_aware": {
            "record_count": len(support),
            "pvc_precision": mean(support, "pvc_precision"),
            "pvc_recall": mean(support, "pvc_recall"),
            "pvc_f1": mean(support, "pvc_f1"),
            "balanced_accuracy": float(np.mean(balanced)) if balanced else None,
        },
        "all_records": {
            "record_count": len(items),
            "specificity": mean(items, "specificity"),
            "brier_score": mean(items, "brier_score"),
            "legacy_zero_filled_macro_pvc_f1": mean(items, "pvc_f1"),
        },
        "mixed_class_records": {
            "record_count": len(both),
            "auroc": mean(both, "auroc"),
            "auprc": mean(both, "auprc"),
        },
    }


def bootstrap_record_macro(per_record: dict[str, dict]) -> dict:
    keys = list(per_record)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    statistics = {
        "support_aware_pvc_precision": [],
        "support_aware_pvc_recall": [],
        "support_aware_pvc_f1": [],
        "all_record_specificity": [],
        "all_record_brier_score": [],
        "mixed_class_auroc": [],
        "mixed_class_auprc": [],
    }
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = [per_record[keys[int(index)]] for index in rng.integers(0, len(keys), len(keys))]
        support = [item for item in sampled if item["has_pvc_reference"]]
        both = [item for item in sampled if item["auroc"] is not None]
        if support:
            statistics["support_aware_pvc_precision"].append(np.mean([item["pvc_precision"] for item in support]))
            statistics["support_aware_pvc_recall"].append(np.mean([item["pvc_recall"] for item in support]))
            statistics["support_aware_pvc_f1"].append(np.mean([item["pvc_f1"] for item in support]))
        statistics["all_record_specificity"].append(np.mean([item["specificity"] for item in sampled]))
        statistics["all_record_brier_score"].append(np.mean([item["brier_score"] for item in sampled]))
        if both:
            statistics["mixed_class_auroc"].append(np.mean([item["auroc"] for item in both]))
            statistics["mixed_class_auprc"].append(np.mean([item["auprc"] for item in both]))
    return {
        name: {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "ci_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }
        for name, values in statistics.items() if values
    }


def flatten_confusion(per_record: dict[str, dict]) -> list[list[int]]:
    matrices = np.asarray([item["confusion_matrix"] for item in per_record.values()], dtype=np.int64)
    return matrices.sum(axis=0).tolist()


def make_per_record_rows(per_record: dict[str, dict]) -> list[dict]:
    rows = []
    for key, metrics in sorted(per_record.items()):
        matrix = metrics["confusion_matrix"]
        rows.append({
            "record_key": key,
            "eligible_N": metrics["eligible_N"],
            "eligible_V": metrics["eligible_V"],
            "has_pvc_reference": metrics["has_pvc_reference"],
            "pvc_precision": metrics["pvc_precision"],
            "pvc_recall": metrics["pvc_recall"],
            "pvc_f1": metrics["pvc_f1"],
            "specificity": metrics["specificity"],
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "brier_score": metrics["brier_score"],
            "false_pvc_decisions": metrics["false_pvc_decisions"],
            "tn": matrix[0][0], "fp": matrix[0][1], "fn": matrix[1][0], "tp": matrix[1][1],
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Experiment 3 artifact: {path}")
    if not rows:
        raise ValueError("Cannot write empty per-record table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dependency_versions() -> dict:
    import sklearn
    import tensorflow as tf

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "wfdb": wfdb.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/gate_s2_int8_tf215_run1/model_int8.tflite"))
    parser.add_argument("--calibration", type=Path, default=Path("models/gate_s2_int8_tf215_run1/calibration.json"))
    parser.add_argument("--freeze-manifest", type=Path, default=Path("results/gate_s2_int8_tf215_run1/manifest.json"))
    parser.add_argument("--record-manifest", type=Path, default=Path("results/experiment0/record_manifest.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--result-dir", type=Path, default=Path("results/experiment3"))
    parser.add_argument("--allow-frozen-svdb-evaluation", action="store_true")
    args = parser.parse_args()

    if not args.allow_frozen_svdb_evaluation:
        parser.error("Experiment 3 requires the explicit --allow-frozen-svdb-evaluation acknowledgement")
    if args.result_dir.exists():
        raise FileExistsError(f"Experiment 3 output already exists: {args.result_dir}")

    model_hash = sha256_file(args.model)
    calibration_hash = sha256_file(args.calibration)
    manifest_hash = sha256_file(args.freeze_manifest)
    if model_hash != EXPECTED_MODEL_SHA256:
        raise ValueError("Frozen int8 model hash does not match Gate S2")
    if calibration_hash != EXPECTED_CALIBRATION_SHA256:
        raise ValueError("Frozen calibration hash does not match Gate S2")
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen Gate S2 manifest hash does not match")

    freeze_manifest = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if freeze_manifest.get("status") != "passed":
        raise ValueError("Gate S2 manifest is not passed")
    if freeze_manifest.get("candidate") != FROZEN_CANDIDATE:
        raise ValueError("Frozen candidate mismatch")
    if freeze_manifest.get("source_condition") != FROZEN_SOURCE_CONDITION:
        raise ValueError("Frozen source-condition mismatch")
    if freeze_manifest.get("lead_profile") != FROZEN_LEAD_PROFILE:
        raise ValueError("Frozen lead-profile mismatch")
    if int(freeze_manifest.get("parameter_count", -1)) != FROZEN_PARAMETER_COUNT:
        raise ValueError("Frozen parameter-count mismatch")
    if float(calibration.get("threshold", -1.0)) != FROZEN_THRESHOLD:
        raise ValueError("Frozen threshold mismatch")
    if calibration.get("method") != "Platt scaling with source- and record-balanced fitting weights":
        raise ValueError("Frozen calibration method mismatch")

    protocol = {
        "experiment": "experiment_3_frozen_svdb_cross_database_evaluation",
        "status": "registered_execution",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": FROZEN_CANDIDATE,
        "artifact": {"path": str(args.model), "sha256": model_hash, "size_bytes": args.model.stat().st_size},
        "calibration": {"path": str(args.calibration), "sha256": calibration_hash, "method": calibration["method"], "coefficient": calibration["coefficient"], "intercept": calibration["intercept"], "threshold": FROZEN_THRESHOLD},
        "freeze_manifest": {"path": str(args.freeze_manifest), "sha256": manifest_hash, "gate": freeze_manifest["gate"], "status": freeze_manifest["status"]},
        "external_source": "svdb",
        "external_role": "locked_external_test",
        "lead_profile": FROZEN_LEAD_PROFILE,
        "preprocessing": "fixed 360 Hz polyphase resampling, 300-sample centered oracle-annotation window, per-window/channel normalization, frozen past-only RR features",
        "r_peak_path": "oracle reference annotation; automatic R-peak comparison is reserved for Experiment 5",
        "prohibited_operations": ["retraining", "seed selection", "architecture selection", "calibration fitting", "threshold selection", "quantization change", "lead-profile change", "preprocessing change"],
        "bootstrap": {"unit": "record", "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "write_once": True,
    }
    args.result_dir.mkdir(parents=True)
    write_json(args.result_dir / "protocol.json", protocol)

    records = load_svdb_records(args.record_manifest)
    scaler = frozen_scaler(calibration)
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    raw_predictions: dict[str, np.ndarray] = {}
    for index, record in enumerate(records, start=1):
        record_base = args.data_root / SOURCE_BY_KEY["svdb"].directory / record["record"]
        accepted, rr = causal_rr_features(record_base, lead_count=2)
        raw = base.tflite_predict(args.model, accepted["waveforms"], rr).astype(np.float32)
        probabilities = core.apply_scaler(scaler, raw).astype(np.float32)
        key = f"svdb:{record['record']}"
        predictions[key] = (accepted["labels"].astype(np.int32), probabilities)
        raw_predictions[key] = raw
        print(f"[{index}/{len(records)}] {key}: {len(raw)} eligible beats", flush=True)

    per_record, summary = core.aggregate_records(predictions, FROZEN_THRESHOLD)
    extended_macro = record_macro_extended(per_record)
    bootstrap = bootstrap_record_macro(per_record)
    rows = make_per_record_rows(per_record)
    pvc_rows = [row for row in rows if row["has_pvc_reference"]]
    worst_decile_count = max(1, int(np.ceil(len(pvc_rows) * 0.10)))
    worst_decile = sorted(pvc_rows, key=lambda row: (row["pvc_f1"], row["pvc_recall"], -row["false_pvc_decisions"], row["record_key"]))[:worst_decile_count]
    error_categories = {
        "true_negative": int(sum(row["tn"] for row in rows)),
        "false_positive": int(sum(row["fp"] for row in rows)),
        "false_negative": int(sum(row["fn"] for row in rows)),
        "true_positive": int(sum(row["tp"] for row in rows)),
        "scope": "oracle-annotation windows only; automatic-R-peak error decomposition is deferred to Experiment 5",
    }

    write_csv(args.result_dir / "per_record_metrics.csv", rows)
    np.savez_compressed(
        args.result_dir / "per_record_predictions.npz",
        **{f"{key.replace(':', '__')}__labels": labels for key, (labels, _) in predictions.items()},
        **{f"{key.replace(':', '__')}__raw_int8_probability": raw_predictions[key] for key in predictions},
        **{f"{key.replace(':', '__')}__calibrated_probability": probabilities for key, (_, probabilities) in predictions.items()},
    )
    result = {
        "experiment": "experiment_3_frozen_svdb_cross_database_evaluation",
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_data_accessed": True,
        "external_source": "svdb",
        "external_test_characterization": "isolated cross-database holdout; not claimed as a pristine blind test because of prior exploratory work",
        "frozen_configuration_verified": True,
        "candidate": FROZEN_CANDIDATE,
        "artifact_sha256": model_hash,
        "calibration_sha256": calibration_hash,
        "freeze_manifest_sha256": manifest_hash,
        "threshold": FROZEN_THRESHOLD,
        "record_count": len(records),
        "pooled": summary["pooled"],
        "record_macro": {**summary["record_macro"], "extended": extended_macro},
        "confusion_matrix": flatten_confusion(per_record),
        "record_level_bootstrap_ci_95": bootstrap,
        "worst_decile_pvc_bearing_records": worst_decile,
        "error_categories": error_categories,
        "artifacts": {"protocol": str(args.result_dir / "protocol.json"), "per_record_metrics": str(args.result_dir / "per_record_metrics.csv"), "per_record_predictions": str(args.result_dir / "per_record_predictions.npz")},
        "dependencies": dependency_versions(),
        "post_evaluation_rule": "No model, threshold, calibration, quantization, preprocessing, lead-profile, or seed change may be selected from these results. Any new hypothesis is exploratory and requires a different future dataset.",
    }
    write_json(args.result_dir / "summary.json", result)
    print(json.dumps({"status": result["status"], "summary": str(args.result_dir / "summary.json"), "record_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
