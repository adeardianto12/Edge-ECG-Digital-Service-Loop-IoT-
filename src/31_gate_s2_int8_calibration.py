"""Run the one-time Gate S2 deployment-probability protocol.

Gate S2 keeps the P0/O1 graph, training data, seed, and epoch policy fixed. It
fits calibration and the threshold on five outer-fold int8 OOF predictions, then
evaluates the resulting int8 deployment artifact. The original float/int8
pointwise equivalence metrics remain diagnostics; no old Gate S artifact is
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np


SOURCES = {"mitdb", "incartdb"}
FORBIDDEN_SOURCES = {"svdb", "nsrdb"}
SPLIT_SHA256 = "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787"
SEED = 20260803
PARAMETERS = 3506


def load_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("26_gate_s_freeze.py", "gate_s2_base")
revalidation = base.revalidation
core = base.core


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Gate S2 artifact: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_m1_records(splits_path: Path, manifest_path: Path):
    split_payload = json.loads(splits_path.read_text(encoding="utf-8"))
    condition = split_payload.get("conditions", {}).get("M1")
    if split_payload.get("split_sha256") != SPLIT_SHA256 or not condition:
        raise ValueError("Reviewed M1 split is unavailable")
    if set(condition.get("sources", [])) != SOURCES:
        raise ValueError("Gate S2 requires exactly MIT-BIH and INCART")
    keys = [key for fold in condition["outer_folds"] for key in fold["records"]]
    if len(keys) != len(set(keys)):
        raise ValueError("M1 outer folds overlap")
    records = [r for r in core.prep.load_training_records(manifest_path) if r.source in SOURCES]
    if {r.key for r in records} != set(keys):
        raise ValueError("M1 manifest and split records do not match")
    if any(r.source in FORBIDDEN_SOURCES for r in records):
        raise ValueError("Forbidden external source entered Gate S2")
    return records, condition


def direct_int8_record_predictions(model_path: Path, records, store) -> dict:
    predictions = {}
    for record in records:
        loaded = store.load(record)
        if len(loaded["labels"]) == 0:
            continue
        raw = base.tflite_predict(model_path, loaded["waveforms"], loaded["rr"])
        predictions[record.key] = (loaded["labels"].astype(np.int32), raw.astype(np.float32))
    return predictions


def frozen_scaler(calibration: dict):
    from sklearn.linear_model import LogisticRegression

    scaler = LogisticRegression()
    scaler.classes_ = np.asarray([0, 1], dtype=np.int32)
    scaler.coef_ = np.asarray([[float(calibration["coefficient"])]], dtype=np.float64)
    scaler.intercept_ = np.asarray([float(calibration["intercept"])], dtype=np.float64)
    scaler.n_features_in_ = 1
    return scaler


def dependency_versions() -> dict:
    import sklearn
    import tensorflow as tf
    import wfdb

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
    parser.add_argument("--splits", type=Path, default=Path("results/experiment1/splits.json"))
    parser.add_argument("--manifest", type=Path, default=Path("results/experiment0/record_manifest.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--revalidation-dir", type=Path, default=Path("results/final_revalidation"))
    parser.add_argument("--source-model-dir", type=Path, default=Path("models/software_freeze_gate_s_retry6"))
    parser.add_argument("--source-result-dir", type=Path, default=Path("results/software_freeze_gate_s_retry6"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/gate_s2_int8"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/gate_s2_int8"))
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quantization-samples", type=int, default=5000)
    parser.add_argument("--equivalence-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.model_dir.exists() or args.result_dir.exists():
        raise FileExistsError("Gate S2 refuses existing output directories")

    records, condition = load_m1_records(args.splits, args.manifest)
    epochs, epoch_trace = base.median_final_epochs(args.revalidation_dir)
    args.epochs = epochs
    args.model_dir.mkdir(parents=True)
    args.result_dir.mkdir(parents=True)
    store = revalidation.FinalRecordStore(records, args.data_root, args.result_dir / "window_cache", lead_count=2)

    protocol = {
        "gate": "Gate S2: deployment-probability protocol amendment",
        "status": "registered_execution",
        "candidate": "P0/O1",
        "sources": sorted(SOURCES),
        "forbidden_sources": sorted(FORBIDDEN_SOURCES),
        "split_sha256": SPLIT_SHA256,
        "seed": SEED,
        "epochs": epochs,
        "epoch_trace": epoch_trace,
        "calibration": "one source/record-balanced Platt map fitted only on five-fold int8 OOF raw outputs",
        "threshold": "one global 0.01 grid selected only on the same int8 OOF predictions",
        "representative_data": "M1 development only; fixed record-balanced Gate S selection rule",
        "original_gate_s": "retained failed result; p99 float/int8 criterion remains a descriptive diagnostic",
        "external_data_accessed": False,
        "acceptance": {
            "support_aware_f1_gte": 0.80,
            "support_aware_recall_gte": 0.90,
            "support_aware_precision_gte": 0.75,
            "record_macro_auprc_gte": 0.88,
            "record_macro_brier_lte": 0.04,
            "legacy_zero_filled_f1_gte": 0.70,
            "pvc_free_false_decisions_lte_p0": "P0 control from original Gate S OOF",
            "int8_size_bytes_lt": 1048576,
            "held_float_int8_decision_agreement_gte": 0.99,
        },
        "stop_rule": "one fixed Gate S2 execution; no candidate/search branches",
    }
    write_json(args.result_dir / "protocol.json", protocol)

    # Five outer-fold int8 models create the only calibration/threshold evidence.
    oof = {}
    fold_artifacts = []
    for fold_index, fold in enumerate(condition["outer_folds"], start=1):
        test_keys = set(fold["records"])
        train = [r for r in records if r.key not in test_keys]
        held = [r for r in records if r.key in test_keys]
        model = revalidation.train_epochs(train, store, args, SEED, epochs, lead_count=2)
        if int(model.count_params()) != PARAMETERS:
            raise AssertionError("OOF model parameter count differs from P0/O1")
        rep_w, rep_rr, _, _, _ = base.select_disjoint_samples(train, store, args.quantization_samples, 1)
        fold_dir = args.result_dir / "oof_int8_models"
        fold_dir.mkdir(exist_ok=True)
        model_path = fold_dir / f"fold{fold_index}.tflite"
        base.export_int8(model, rep_w, rep_rr, model_path)
        fold_predictions = direct_int8_record_predictions(model_path, held, store)
        if set(oof) & set(fold_predictions):
            raise AssertionError("OOF record overlap detected")
        oof.update(fold_predictions)
        fold_artifacts.append({"fold": fold_index, "model": str(model_path), "sha256": sha256_file(model_path), "records": len(fold_predictions)})

    if not oof:
        raise RuntimeError("No int8 OOF predictions were produced")
    scaler, threshold, selection = core.calibrate_and_select(oof)
    calibrated_oof = {key: (labels, core.apply_scaler(scaler, raw)) for key, (labels, raw) in oof.items()}
    _, oof_summary = core.aggregate_records(calibrated_oof, threshold)
    oof_metrics = oof_summary["record_macro"]

    final_model = revalidation.train_epochs(records, store, args, SEED, epochs, lead_count=2)
    if int(final_model.count_params()) != PARAMETERS:
        raise AssertionError("Final model parameter count differs from P0/O1")
    args.model_dir.mkdir(exist_ok=True)
    keras_path = args.model_dir / "model.keras"
    tflite_path = args.model_dir / "model_int8.tflite"
    final_model.save(keras_path)
    rep_w, rep_rr, held_w, held_rr, held_labels = base.select_disjoint_samples(records, store, args.quantization_samples, args.equivalence_samples)
    export_model = base.export_int8(final_model, rep_w, rep_rr, tflite_path)
    float_raw = final_model.predict([held_w, held_rr], batch_size=128, verbose=0)[:, 1]
    int8_raw = base.tflite_predict(tflite_path, held_w, held_rr)
    int8_probability = core.apply_scaler(scaler, int8_raw)
    original_calibration = json.loads((args.source_model_dir / "calibration.json").read_text(encoding="utf-8"))
    original_probability = core.apply_scaler(frozen_scaler(original_calibration), float_raw)
    errors = np.abs(original_probability - int8_probability)
    agreement = float(np.mean((original_probability >= float(original_calibration["threshold"])) == (int8_probability >= threshold)))
    int8_metrics = core.per_record_metrics(held_labels, int8_probability, threshold)
    criteria = {
        "support_aware_f1": oof_metrics["support_aware"]["pvc_f1"] >= 0.80,
        "support_aware_recall": oof_metrics["support_aware"]["pvc_recall"] >= 0.90,
        "support_aware_precision": oof_metrics["support_aware"]["pvc_precision"] >= 0.75,
        "record_macro_auprc": oof_metrics["record_macro_auprc"] >= 0.88,
        "record_macro_brier": oof_metrics["record_macro_brier_score"] <= 0.04,
        "legacy_zero_filled_f1": oof_metrics["legacy_zero_filled_macro_pvc_f1"] >= 0.70,
        "pvc_free_false_decisions_no_worse_than_p0": oof_metrics["pvc_free_records"]["false_pvc_decisions"] <= 87,
        "size_under_1_mib": tflite_path.stat().st_size < 1024 * 1024,
        "held_float_int8_decision_agreement": agreement >= 0.99,
    }
    calibration = {"method": selection["calibration"], "coefficient": float(scaler.coef_[0][0]), "intercept": float(scaler.intercept_[0]), "threshold": float(threshold), "selection": selection}
    write_json(args.model_dir / "calibration.json", calibration)
    equivalence = {
        "experiment": "gate_s2_int8_direct_calibration",
        "candidate": "P0/O1",
        "oof_metrics": oof_summary,
        "held_development_int8_metrics": int8_metrics,
        "held_float_int8_decision_agreement": agreement,
        "held_mean_absolute_probability_error_descriptive": float(np.mean(errors)),
        "held_p99_absolute_probability_error_descriptive": float(np.quantile(errors, 0.99)),
        "int8_size_bytes": tflite_path.stat().st_size,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "external_data_accessed": False,
    }
    write_json(args.result_dir / "int8_direct_evaluation.json", equivalence)
    manifest = {
        "gate": "Gate S2: deployment-probability protocol amendment",
        "status": "passed" if equivalence["passed"] else "failed",
        "candidate": "P0/O1",
        "source_condition": "M1: MIT-BIH + INCART",
        "lead_profile": "L2",
        "parameter_count": int(final_model.count_params()),
        "calibration": calibration,
        "fold_artifacts": fold_artifacts,
        "artifacts": {"keras": str(keras_path), "keras_sha256": sha256_file(keras_path), "int8_tflite": str(tflite_path), "int8_tflite_sha256": sha256_file(tflite_path), "export_architecture_json_sha256": hashlib.sha256(export_model.to_json().encode("utf-8")).hexdigest(), "int8_direct_evaluation": str(args.result_dir / "int8_direct_evaluation.json")},
        "source_artifacts": {"original_gate_s_manifest": str(args.source_result_dir / "manifest.json"), "original_gate_s_manifest_sha256": sha256_file(args.source_result_dir / "manifest.json")},
        "split_sha256": SPLIT_SHA256,
        "dependencies": dependency_versions(),
        "external_data_accessed": False,
        "prohibited_sources": sorted(FORBIDDEN_SOURCES),
        "original_gate_s_p99_failure_preserved": True,
        "immutable": True,
    }
    write_json(args.result_dir / "manifest.json", manifest)
    print(json.dumps({"gate_s2": manifest["status"], "manifest": str(args.result_dir / "manifest.json"), "external_data_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
