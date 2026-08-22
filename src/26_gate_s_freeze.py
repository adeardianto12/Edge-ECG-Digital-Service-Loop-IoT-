"""Create the immutable Gate S software-freeze artifacts for retained M1/L2 P0/O1.

This command is intentionally isolated from the historical freeze prototype.  It
requires the complete final architecture revalidation, trains no candidate other
than the retained M1/L2 P0/O1 model, and refuses to overwrite a freeze attempt.
Only MIT-BIH and INCART development records are ever loaded.
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


SPLIT_SHA256 = "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787"
SOURCES = {"mitdb", "incartdb"}
FORBIDDEN_SOURCES = {"nsrdb", "svdb"}
CANONICAL_SEED = 20260803
EXPECTED_REVALIDATION_RUNS = 60
EXPECTED_PARAMETER_COUNT = 3506


def load_revalidation_module():
    path = Path(__file__).with_name("24_final_revalidate.py")
    spec = importlib.util.spec_from_file_location("gate_s_revalidation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load final-revalidation implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


revalidation = load_revalidation_module()
core = revalidation.core


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Gate S artifact: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def require_empty_targets(model_dir: Path, result_dir: Path) -> None:
    for directory in (model_dir, result_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Gate S target already contains evidence: {directory}")
        if directory.exists() and not directory.is_dir():
            raise FileExistsError(f"Gate S target is not a directory: {directory}")


def validate_revalidation(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment") != "final_architecture_m0_m1_m2_l1_l2_revalidation":
        raise ValueError("Gate S requires the final-architecture revalidation summary")
    if summary.get("status") != "complete" or summary.get("candidate") != "P0/O1":
        raise ValueError("Final revalidation is incomplete or does not retain P0/O1")
    if summary.get("split_sha256") != SPLIT_SHA256 or summary.get("external_data_accessed"):
        raise ValueError("Final revalidation provenance is invalid")
    conditions = summary.get("conditions", {})
    if sum(int(value.get("run_count", 0)) for value in conditions.values()) != EXPECTED_REVALIDATION_RUNS:
        raise ValueError("Final revalidation does not contain all 60 required runs")
    required = {"M0_L2", "M1_L2", "M2_L2", "M1_L1"}
    if set(conditions) != required:
        raise ValueError("Final revalidation conditions are incomplete")
    for name in required:
        condition = conditions[name]
        if condition.get("run_count") != 15 or condition.get("expected_run_count") != 15:
            raise ValueError(f"Final revalidation condition is incomplete: {name}")
    m1 = conditions["M1_L2"]
    if m1.get("source_condition") != "M1" or m1.get("lead_profile") != "L2":
        raise ValueError("M1/L2 is not the retained source and lead profile")
    if m1.get("parameter_count") != [EXPECTED_PARAMETER_COUNT]:
        raise ValueError("Retained parameter count does not match P0/O1")
    comparisons = {item["comparison"]: item for item in summary.get("paired_record_comparisons", [])}
    lead_ci = comparisons.get("M1_L2_minus_M1_L1", {}).get("bootstrap", {}).get("ci_95")
    m2_ci = comparisons.get("M2_L2_minus_M1_L2", {}).get("bootstrap", {}).get("ci_95")
    m0_ci = comparisons.get("M1_L2_minus_M0_L2", {}).get("bootstrap", {}).get("ci_95")
    if not (lead_ci and m2_ci and m0_ci and lead_ci[0] > 0 and m2_ci[1] < 0 and m0_ci[0] <= 0 <= m0_ci[1]):
        raise ValueError("Revalidation evidence does not support the retained M1/L2 decision")
    return summary


def median_final_epochs(revalidation_dir: Path) -> tuple[int, list[int]]:
    values = []
    for path in sorted(revalidation_dir.glob("M1_L2_fold*_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("split_sha256") != SPLIT_SHA256 or payload.get("external_data_accessed"):
            raise ValueError(f"Invalid M1/L2 revalidation artifact: {path}")
        values.append(int(payload["training"]["final_epochs"]))
    if len(values) != 15:
        raise ValueError("Gate S requires 15 retained M1/L2 revalidation runs")
    return int(np.median(values)), values


def select_disjoint_samples(records, store, quantization_limit: int, equivalence_limit: int):
    """Create deterministic record-balanced quantization and held equivalence sets."""
    rng = np.random.default_rng(20260811)
    representative, held = [], []
    for record in records:
        loaded = store.load(record)
        size = len(loaded["labels"])
        if not size:
            continue
        order = rng.permutation(size)
        rep_count = min(64, size)
        held_count = min(64, max(size - rep_count, 0))
        amplitude = np.max(np.abs(loaded["waveforms"]), axis=(1, 2))
        rr_magnitude = np.max(np.abs(loaded["rr"]), axis=1)
        extreme = np.concatenate([
            np.argsort(amplitude)[-16:],
            np.argsort(rr_magnitude)[-8:],
        ])
        rep_order = list(dict.fromkeys([int(index) for index in extreme] + [int(index) for index in order]))
        for index in rep_order[:rep_count]:
            representative.append((loaded["waveforms"][index], loaded["rr"][index]))
        rep_indices = set(rep_order[:rep_count])
        held_order = [int(index) for index in order if int(index) not in rep_indices]
        for index in held_order[:held_count]:
            held.append((loaded["waveforms"][index], loaded["rr"][index], loaded["labels"][index]))
        if len(representative) >= quantization_limit and len(held) >= equivalence_limit:
            break
    if not representative or not held:
        raise ValueError("Development samples are unavailable for int8 validation")
    representative = representative[:quantization_limit]
    held = held[:equivalence_limit]
    waveforms = np.asarray([item[0] for item in representative], dtype=np.float32)
    rr = np.asarray([item[1] for item in representative], dtype=np.float32)
    held_waveforms = np.asarray([item[0] for item in held], dtype=np.float32)
    held_rr = np.asarray([item[1] for item in held], dtype=np.float32)
    held_labels = np.asarray([item[2] for item in held], dtype=np.int32)
    return waveforms, rr, held_waveforms, held_rr, held_labels


def quantize(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    dtype = np.dtype(detail["dtype"])
    if scale == 0:
        return values.astype(dtype)
    info = np.iinfo(dtype)
    return np.clip(np.rint(values / scale + zero_point), info.min, info.max).astype(dtype)


def dequantize(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    return (values.astype(np.float32) - zero_point) * scale


def tflite_predict(model_path: Path, waveforms: np.ndarray, rr: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    values = []
    for start in range(len(waveforms)):
        end = start + 1
        details = interpreter.get_input_details()
        waveform = next((item for item in details if tuple(item["shape"][1:]) == (300, 2)), None)
        rr_input = next((item for item in details if tuple(item["shape"][1:]) == (5,)), None)
        if waveform is None or rr_input is None or len(interpreter.get_output_details()) != 1:
            raise ValueError("Unexpected full-int8 TFLite input/output contract")
        interpreter.set_tensor(waveform["index"], quantize(waveforms[start:end], waveform))
        interpreter.set_tensor(rr_input["index"], quantize(rr[start:end], rr_input))
        interpreter.invoke()
        output = interpreter.get_output_details()[0]
        values.append(dequantize(interpreter.get_tensor(output["index"]), output))
    return np.concatenate(values, axis=0)[:, 1]


def build_fixed_batch_export_model(model):
    """Recreate the identical P0/O1 graph with a single-beat TFLite signature.

    TensorFlow 2.21's integer calibrator fails on this Conv1D graph when the
    batch dimension is dynamic. A fixed batch of one is the deployed beat-wise
    inference contract and does not alter weights, parameters, or operations.
    """
    import tensorflow as tf

    waveform = tf.keras.Input(batch_shape=(1, 300, 2), name="ecg_beat")
    morphology = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu")(waveform)
    morphology = tf.keras.layers.MaxPooling1D(2)(morphology)
    morphology = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(morphology)
    morphology = tf.keras.layers.GlobalAveragePooling1D()(morphology)
    rr_inputs = tf.keras.Input(batch_shape=(1, 5), name="causal_rr")
    rr_features = tf.keras.layers.Dense(8, activation="relu")(rr_inputs)
    fused = tf.keras.layers.Concatenate(name="fusion")([morphology, rr_features])
    hidden = tf.keras.layers.Dense(16, activation="relu")(fused)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(hidden)
    fixed = tf.keras.Model([waveform, rr_inputs], outputs, name="p0_o1_int8_export_batch1")
    fixed.set_weights(model.get_weights())
    if fixed.count_params() != model.count_params():
        raise AssertionError("Fixed-batch export changed parameter count")
    return fixed


def export_int8(model, representative_waveforms: np.ndarray, representative_rr: np.ndarray, path: Path):
    import tensorflow as tf

    fixed = build_fixed_batch_export_model(model)
    converter = tf.lite.TFLiteConverter.from_keras_model(fixed)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for index in range(len(representative_waveforms)):
            yield {"ecg_beat": representative_waveforms[index:index + 1], "causal_rr": representative_rr[index:index + 1]}

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    path.write_bytes(converter.convert())
    return fixed


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
    parser.add_argument("--revalidation-dir", type=Path, default=Path("results/final_revalidation"))
    parser.add_argument("--manifest", type=Path, default=Path("results/experiment0/record_manifest.csv"))
    parser.add_argument("--splits", type=Path, default=Path("results/experiment1/splits.json"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--model-dir", type=Path, default=Path("models/software_freeze"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/software_freeze"))
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quantization-samples", type=int, default=5000)
    parser.add_argument("--equivalence-samples", type=int, default=1000)
    args = parser.parse_args()

    if args.steps_per_epoch < 1 or args.batch_size < 2:
        parser.error("Steps per epoch and batch size must be positive")
    if args.quantization_samples < 1 or args.equivalence_samples < 1:
        parser.error("Quantization and equivalence sample counts must be positive")
    require_empty_targets(args.model_dir, args.result_dir)
    revalidation_summary = validate_revalidation(args.revalidation_dir / "summary.json")
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    m1 = split_payload.get("conditions", {}).get("M1")
    if split_payload.get("split_sha256") != SPLIT_SHA256 or not m1 or set(m1.get("sources", [])) != SOURCES:
        raise ValueError("Gate S requires the reviewed M1 MIT-BIH + INCART split")
    epochs, revalidation_epochs = median_final_epochs(args.revalidation_dir)
    records = [record for record in core.prep.load_training_records(args.manifest) if record.source in SOURCES]
    if {record.source for record in records} != SOURCES:
        raise ValueError("Gate S development source boundary is incomplete")
    if any(record.source in FORBIDDEN_SOURCES for record in records):
        raise ValueError("Gate S source boundary includes a prohibited source")

    args.model_dir.mkdir(parents=True, exist_ok=False)
    args.result_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = args.result_dir / "window_cache"
    store = revalidation.FinalRecordStore(records, args.data_root, cache_dir, lead_count=2)
    protocol = {
        "gate": "Gate S: Software Evaluation Freeze",
        "candidate": "P0/O1",
        "source_condition": "M1",
        "sources": sorted(SOURCES),
        "lead_profile": "L2",
        "seed": CANONICAL_SEED,
        "split_sha256": SPLIT_SHA256,
        "epoch_policy": {
            "method": "global median of all 15 completed M1/L2 final-revalidation fits",
            "selected_epochs": epochs,
            "revalidation_final_epochs": revalidation_epochs,
        },
        "calibration_and_threshold": "one Platt calibration and one global threshold from five-fold M1 OOF raw probabilities",
        "external_data_accessed": False,
        "prohibited_sources": sorted(FORBIDDEN_SOURCES),
        "overwrite_policy": "refuse existing non-empty model or result target",
    }
    write_json(args.result_dir / "protocol.json", protocol)

    # Each OOF model is restricted to four development folds; no outer-fold record is seen in its fit.
    oof = {}
    for fold_index, fold in enumerate(m1["outer_folds"], start=1):
        test_keys = set(fold["records"])
        train = [record for record in records if record.key not in test_keys]
        held = [record for record in records if record.key in test_keys]
        if not train or not held or {record.key for record in train} & {record.key for record in held}:
            raise AssertionError(f"OOF record isolation failed for fold {fold_index}")
        model = revalidation.train_epochs(train, store, args, CANONICAL_SEED, epochs, lead_count=2)
        held_predictions = core.predict_records(model, "O1", held, store)
        overlap = set(oof) & set(held_predictions)
        if overlap:
            raise AssertionError(f"Duplicate OOF records: {sorted(overlap)[:3]}")
        oof.update(held_predictions)
    eligible_record_keys = {record.key for record in records if len(store.load(record)["labels"])}
    if set(oof) != eligible_record_keys:
        raise AssertionError("OOF coverage is not exactly the M1 records with eligible N/V windows")
    scaler, threshold, selection = core.calibrate_and_select(oof)
    calibrated_oof = {key: (labels, core.apply_scaler(scaler, raw)) for key, (labels, raw) in oof.items()}
    oof_per_record, oof_summary = core.aggregate_records(calibrated_oof, threshold)
    oof_dir = args.result_dir / "oof_predictions"
    oof_dir.mkdir()
    for key, (labels, raw) in oof.items():
        path = oof_dir / f"{key.replace(':', '_')}.npz"
        np.savez_compressed(path, labels=labels, raw_probability=raw, calibrated_probability=calibrated_oof[key][1])

    final_model = revalidation.train_epochs(records, store, args, CANONICAL_SEED, epochs, lead_count=2)
    if int(final_model.count_params()) != EXPECTED_PARAMETER_COUNT:
        raise AssertionError("Final model parameter count differs from retained P0/O1")
    keras_path = args.model_dir / "model.keras"
    tflite_path = args.model_dir / "model_int8.tflite"
    final_model.save(keras_path)
    rep_waveforms, rep_rr, held_waveforms, held_rr, held_labels = select_disjoint_samples(
        records, store, args.quantization_samples, args.equivalence_samples
    )
    int8_export_model = export_int8(final_model, rep_waveforms, rep_rr, tflite_path)
    float_raw = final_model.predict([held_waveforms, held_rr], batch_size=128, verbose=0)[:, 1]
    int8_raw = tflite_predict(tflite_path, held_waveforms, held_rr)
    float_probability = core.apply_scaler(scaler, float_raw)
    int8_probability = core.apply_scaler(scaler, int8_raw)
    float_per_record = {"held_development_equivalence": core.per_record_metrics(held_labels, float_probability, threshold)}
    int8_per_record = {"held_development_equivalence": core.per_record_metrics(held_labels, int8_probability, threshold)}
    float_metrics = float_per_record["held_development_equivalence"]
    int8_metrics = int8_per_record["held_development_equivalence"]
    errors = np.abs(float_probability - int8_probability)
    decision_agreement = float(np.mean((float_probability >= threshold) == (int8_probability >= threshold)))
    criteria = {
        "size_under_1_mib": tflite_path.stat().st_size < 1024 * 1024,
        "decision_agreement_at_least_0_99": decision_agreement >= 0.99,
        "mean_error_at_most_0_02": float(np.mean(errors)) <= 0.02,
        "p99_error_at_most_0_05": float(np.quantile(errors, 0.99)) <= 0.05,
        "support_aware_f1_loss_at_most_0_01": float_metrics["pvc_f1"] - int8_metrics["pvc_f1"] <= 0.01,
        "recall_loss_at_most_0_01": float_metrics["pvc_recall"] - int8_metrics["pvc_recall"] <= 0.01,
    }
    equivalence = {
        "experiment": "gate_s_int8_equivalence",
        "candidate": "P0/O1",
        "equivalence_set": {
            "source": "M1 development only; held disjoint from representative quantization samples",
            "samples": int(len(held_labels)),
            "quantization_samples": int(len(rep_waveforms)),
            "sampling_seed": 20260811,
        },
        "threshold": threshold,
        "float_int8_decision_agreement": decision_agreement,
        "mean_absolute_calibrated_probability_error": float(np.mean(errors)),
        "p99_absolute_calibrated_probability_error": float(np.quantile(errors, 0.99)),
        "int8_size_bytes": tflite_path.stat().st_size,
        "float_metrics": float_metrics,
        "int8_metrics": int8_metrics,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }
    write_json(args.result_dir / "int8_equivalence.json", equivalence)
    calibration = {
        "method": selection["calibration"],
        "coefficient": float(scaler.coef_[0][0]),
        "intercept": float(scaler.intercept_[0]),
        "threshold": threshold,
        "selection": selection,
    }
    write_json(args.model_dir / "calibration.json", calibration)
    write_json(args.result_dir / "oof_summary.json", {"outer_folds": 5, "record_count": len(oof), "excluded_zero_eligible_record_count": len(records) - len(oof), "metrics": oof_summary, "per_record": oof_per_record})
    model_json = final_model.to_json()
    manifest = {
        "gate": "Gate S: Software Evaluation Freeze",
        "status": "passed" if equivalence["passed"] else "failed",
        "candidate": "P0/O1",
        "source_condition": "M1: MIT-BIH + INCART",
        "lead_profile": "L2",
        "input_contract": {"waveform": [300, 2], "causal_rr": [5]},
        "int8_input_contract": {"batch": 1, "waveform": [1, 300, 2], "causal_rr": [1, 5]},
        "preprocessing": {"canonical_sample_rate_hz": 360, "window_samples": 300, "window_center": "R peak", "normalization": "per-window per-channel zero mean/unit standard deviation", "rr_features": "past-only causal 5-vector"},
        "seed_policy": {"canonical_final_training_seed": CANONICAL_SEED, "oof_training_seed": CANONICAL_SEED},
        "epochs": protocol["epoch_policy"],
        "calibration": {key: calibration[key] for key in ("method", "coefficient", "intercept", "threshold")},
        "oof": {"folds": 5, "record_count": len(oof), "selection_metric": "feasible pooled and support-aware recall >= 0.90, then support-aware record-macro F1"},
        "revalidation": {"summary": str(args.revalidation_dir / "summary.json"), "summary_sha256": sha256_file(args.revalidation_dir / "summary.json"), "run_count": EXPECTED_REVALIDATION_RUNS, "retained_condition": "M1_L2"},
        "artifacts": {"keras": str(keras_path), "keras_sha256": sha256_file(keras_path), "int8_tflite": str(tflite_path), "int8_tflite_sha256": sha256_file(tflite_path), "architecture_json_sha256": hashlib.sha256(model_json.encode("utf-8")).hexdigest(), "int8_export_architecture_json_sha256": hashlib.sha256(int8_export_model.to_json().encode("utf-8")).hexdigest(), "parameter_count": int(final_model.count_params())},
        "inputs": {"record_manifest_sha256": sha256_file(args.manifest), "splits_sha256": sha256_file(args.splits)},
        "code": {"gate_s_sha256": sha256_file(Path(__file__)), "revalidation_sha256": sha256_file(Path(__file__).with_name("24_final_revalidate.py")), "core_sha256": sha256_file(Path(__file__).with_name("14_experiment2_5_optimize.py")), "preprocessing_sha256": sha256_file(Path(__file__).with_name("multisource_ecg.py"))},
        "dependencies": dependency_versions(),
        "int8_equivalence": {"path": str(args.result_dir / "int8_equivalence.json"), "sha256": sha256_file(args.result_dir / "int8_equivalence.json"), "passed": equivalence["passed"]},
        "external_data_accessed": False,
        "prohibited_sources": sorted(FORBIDDEN_SOURCES),
        "pending_deployment_evidence": ["pending_experiment7: gateway latency", "pending_experiment7: gateway memory", "pending_gate_h: endpoint transport", "pending_gate_h: energy"],
        "immutable": True,
    }
    write_json(args.result_dir / "manifest.json", manifest)
    print(json.dumps({"gate_s": manifest["status"], "manifest": str(args.result_dir / "manifest.json"), "int8_equivalence": equivalence["passed"], "external_data_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
