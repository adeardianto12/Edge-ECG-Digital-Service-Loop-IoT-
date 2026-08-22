"""Development-only fixed-range QAT remediation for the failed Gate S p99 gate.

The P0/O1 topology, parameter count, data boundary, split, seed, calibration,
and threshold rule remain fixed. Fake-quantization operators add no trainable
parameters or inference branch; they make development training aware of the
declared int8 ranges. Existing freeze evidence is never overwritten.
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
SEED = 20260803
PARAMETERS = 3506


def load_freeze_module():
    path = Path(__file__).with_name("26_gate_s_freeze.py")
    spec = importlib.util.spec_from_file_location("gate_s_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Gate S base implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_freeze_module()
core = base.core


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite QAT artifact: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def require_empty(*directories: Path) -> None:
    for directory in directories:
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise FileExistsError(f"QAT target already contains evidence: {directory}")


def fake_quant(tf, value, minimum: float, maximum: float, name: str):
    return tf.keras.layers.Lambda(
        lambda tensor: tf.quantization.fake_quant_with_min_max_args(
            tensor, min=minimum, max=maximum, num_bits=8, narrow_range=False
        ),
        name=name,
    )(value)


def build_qat_model(batch_one: bool = False):
    import tensorflow as tf

    shape_kwargs = {"batch_shape": (1, 300, 2)} if batch_one else {"shape": (300, 2)}
    waveform = tf.keras.Input(**shape_kwargs, name="ecg_beat")
    waveform_q = fake_quant(tf, waveform, -8.0, 8.0, "fq_waveform_input")
    morphology = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu")(waveform_q)
    morphology = fake_quant(tf, morphology, 0.0, 8.0, "fq_morphology_1")
    morphology = tf.keras.layers.MaxPooling1D(2)(morphology)
    morphology = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(morphology)
    morphology = fake_quant(tf, morphology, 0.0, 8.0, "fq_morphology_2")
    morphology = tf.keras.layers.GlobalAveragePooling1D()(morphology)
    morphology = fake_quant(tf, morphology, 0.0, 8.0, "fq_morphology_pool")
    rr_kwargs = {"batch_shape": (1, 5)} if batch_one else {"shape": (5,)}
    rr_inputs = tf.keras.Input(**rr_kwargs, name="causal_rr")
    rr_inputs_q = fake_quant(tf, rr_inputs, -8.0, 8.0, "fq_rr_input")
    rr_features = tf.keras.layers.Dense(8, activation="relu")(rr_inputs_q)
    rr_features = fake_quant(tf, rr_features, 0.0, 8.0, "fq_rr_features")
    fused = tf.keras.layers.Concatenate(name="fusion")([morphology, rr_features])
    fused = fake_quant(tf, fused, 0.0, 8.0, "fq_fusion")
    hidden = tf.keras.layers.Dense(16, activation="relu")(fused)
    hidden = fake_quant(tf, hidden, 0.0, 8.0, "fq_classifier_hidden")
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="class")(hidden)
    return tf.keras.Model([waveform, rr_inputs], outputs, name="gate_s_qat_p0_o1")


def train_qat(records, store, args, seed: int, epochs: int):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    sampler = core.HierarchicalSampler(records, store, seed, augment=False)
    model = build_qat_model()
    if model.count_params() != PARAMETERS:
        raise AssertionError(f"QAT parameter count changed: {model.count_params()}")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    for _ in range(epochs):
        for _ in range(args.steps_per_epoch):
            waveforms, rr, labels = sampler.sample(args.batch_size)
            model.train_on_batch([waveforms, rr], labels)
    return model


def export_int8(model, representative_waveforms, representative_rr, path: Path):
    import tensorflow as tf

    fixed = build_qat_model(batch_one=True)
    fixed.set_weights(model.get_weights())
    if fixed.count_params() != PARAMETERS:
        raise AssertionError("QAT fixed-batch export changed parameter count")
    converter = tf.lite.TFLiteConverter.from_keras_model(fixed)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for index in range(len(representative_waveforms)):
            yield {
                "ecg_beat": representative_waveforms[index:index + 1],
                "causal_rr": representative_rr[index:index + 1],
            }

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    path.write_bytes(converter.convert())
    return fixed


def validate_protocol(revalidation_dir: Path, protocol_path: Path) -> dict:
    qat_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if qat_protocol.get("status") != "registered":
        raise ValueError("QAT protocol is not registered")
    if qat_protocol.get("split_sha256") != SPLIT_SHA256 or set(qat_protocol.get("data_sources", [])) != SOURCES:
        raise ValueError("QAT protocol provenance boundary is invalid")
    summary = json.loads((revalidation_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("candidate") != "P0/O1" or summary.get("external_data_accessed"):
        raise ValueError("QAT requires complete, external-data-free P0/O1 revalidation")
    return qat_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("results/gate_s_qat/protocol.json"))
    parser.add_argument("--revalidation-dir", type=Path, default=Path("results/final_revalidation"))
    parser.add_argument("--manifest", type=Path, default=Path("results/experiment0/record_manifest.csv"))
    parser.add_argument("--splits", type=Path, default=Path("results/experiment1/splits.json"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--model-dir", type=Path, default=Path("models/gate_s_qat_retry1"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/gate_s_qat_retry1"))
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=19)
    parser.add_argument("--quantization-samples", type=int, default=5000)
    parser.add_argument("--equivalence-samples", type=int, default=1000)
    args = parser.parse_args()
    require_empty(args.model_dir, args.result_dir)
    protocol = validate_protocol(args.revalidation_dir, args.protocol)
    if args.epochs != protocol["epochs"] or args.steps_per_epoch != protocol["steps_per_epoch"]:
        raise ValueError("Execution budget differs from registered QAT protocol")
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    condition = split_payload.get("conditions", {}).get("M1")
    if split_payload.get("split_sha256") != SPLIT_SHA256 or not condition or set(condition.get("sources", [])) != SOURCES:
        raise ValueError("QAT requires the reviewed M1 split")
    records = [record for record in core.prep.load_training_records(args.manifest) if record.source in SOURCES]
    if {record.source for record in records} != SOURCES or any(record.source in FORBIDDEN_SOURCES for record in records):
        raise ValueError("QAT source boundary failed")
    args.model_dir.mkdir(parents=True, exist_ok=False)
    args.result_dir.mkdir(parents=True, exist_ok=False)
    store = base.revalidation.FinalRecordStore(records, args.data_root, args.result_dir / "window_cache", lead_count=2)
    write_json(args.result_dir / "protocol.json", protocol | {"status": "executed", "code_sha256": sha256_file(Path(__file__))})

    oof = {}
    for fold_index, fold in enumerate(condition["outer_folds"], start=1):
        test_keys = set(fold["records"])
        train = [record for record in records if record.key not in test_keys]
        test = [record for record in records if record.key in test_keys]
        model = train_qat(train, store, args, SEED, args.epochs)
        predictions = core.predict_records(model, "O1", test, store)
        if set(oof) & set(predictions):
            raise AssertionError(f"Duplicate QAT OOF records in fold {fold_index}")
        oof.update(predictions)
    eligible = {record.key for record in records if len(store.load(record)["labels"])}
    if set(oof) != eligible:
        raise AssertionError("QAT OOF coverage does not match eligible M1 records")
    scaler, threshold, selection = core.calibrate_and_select(oof)
    calibrated_oof = {key: (labels, core.apply_scaler(scaler, raw)) for key, (labels, raw) in oof.items()}
    _, oof_summary = core.aggregate_records(calibrated_oof, threshold)
    oof_dir = args.result_dir / "oof_predictions"
    oof_dir.mkdir()
    for key, (labels, raw) in oof.items():
        np.savez_compressed(oof_dir / f"{key.replace(':', '_')}.npz", labels=labels, raw_probability=raw, calibrated_probability=calibrated_oof[key][1])

    final_model = train_qat(records, store, args, SEED, args.epochs)
    representative_waveforms, representative_rr, held_waveforms, held_rr, held_labels = base.select_disjoint_samples(records, store, args.quantization_samples, args.equivalence_samples)
    keras_path = args.model_dir / "model_qat.keras"
    tflite_path = args.model_dir / "model_qat_int8.tflite"
    final_model.save(keras_path)
    export_model = export_int8(final_model, representative_waveforms, representative_rr, tflite_path)
    float_raw = final_model.predict([held_waveforms, held_rr], batch_size=128, verbose=0)[:, 1]
    int8_raw = base.tflite_predict(tflite_path, held_waveforms, held_rr)
    float_probability = core.apply_scaler(scaler, float_raw)
    int8_probability = core.apply_scaler(scaler, int8_raw)
    float_metrics = core.per_record_metrics(held_labels, float_probability, threshold)
    int8_metrics = core.per_record_metrics(held_labels, int8_probability, threshold)
    errors = np.abs(float_probability - int8_probability)
    agreement = float(np.mean((float_probability >= threshold) == (int8_probability >= threshold)))
    criteria = {
        "size_under_1_mib": tflite_path.stat().st_size < 1024 * 1024,
        "decision_agreement_at_least_0_99": agreement >= 0.99,
        "mean_error_at_most_0_02": float(np.mean(errors)) <= 0.02,
        "p99_error_at_most_0_05": float(np.quantile(errors, 0.99)) <= 0.05,
        "f1_loss_at_most_0_01": float_metrics["pvc_f1"] - int8_metrics["pvc_f1"] <= 0.01,
        "recall_loss_at_most_0_01": float_metrics["pvc_recall"] - int8_metrics["pvc_recall"] <= 0.01,
    }
    equivalence = {
        "experiment": "gate_s_qat_int8_equivalence",
        "candidate": "P0/O1 with registered fixed-range QAT remediation",
        "samples": int(len(held_labels)),
        "quantization_samples": int(len(representative_waveforms)),
        "threshold": threshold,
        "float_int8_decision_agreement": agreement,
        "mean_absolute_calibrated_probability_error": float(np.mean(errors)),
        "p99_absolute_calibrated_probability_error": float(np.quantile(errors, 0.99)),
        "int8_size_bytes": tflite_path.stat().st_size,
        "float_metrics": float_metrics,
        "int8_metrics": int8_metrics,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "external_data_accessed": False,
    }
    write_json(args.result_dir / "int8_equivalence.json", equivalence)
    calibration = {"method": selection["calibration"], "coefficient": float(scaler.coef_[0][0]), "intercept": float(scaler.intercept_[0]), "threshold": threshold, "selection": selection}
    write_json(args.model_dir / "calibration.json", calibration)
    write_json(args.result_dir / "oof_summary.json", {"folds": 5, "record_count": len(oof), "metrics": oof_summary})
    manifest = {
        "gate": "Gate S: Software Evaluation Freeze",
        "status": "passed" if equivalence["passed"] else "failed",
        "remediation": "fixed-range fake-quant QAT; architecture and parameter count unchanged",
        "candidate": "P0/O1",
        "source_condition": "M1: MIT-BIH + INCART",
        "lead_profile": "L2",
        "parameter_count": int(final_model.count_params()),
        "seed": SEED,
        "epochs": args.epochs,
        "split_sha256": SPLIT_SHA256,
        "calibration": {key: calibration[key] for key in ("method", "coefficient", "intercept", "threshold")},
        "artifacts": {"keras": str(keras_path), "keras_sha256": sha256_file(keras_path), "int8_tflite": str(tflite_path), "int8_tflite_sha256": sha256_file(tflite_path), "qat_export_architecture_json_sha256": hashlib.sha256(export_model.to_json().encode("utf-8")).hexdigest()},
        "code_sha256": {"qat": sha256_file(Path(__file__)), "base": sha256_file(Path(__file__).with_name("26_gate_s_freeze.py")), "core": sha256_file(Path(__file__).with_name("14_experiment2_5_optimize.py"))},
        "external_data_accessed": False,
        "prohibited_sources": sorted(FORBIDDEN_SOURCES),
        "int8_equivalence": {"path": str(args.result_dir / "int8_equivalence.json"), "sha256": sha256_file(args.result_dir / "int8_equivalence.json"), "passed": equivalence["passed"]},
        "pending_deployment_evidence": ["pending_experiment7", "pending_gate_h"],
        "immutable": True,
    }
    write_json(args.result_dir / "manifest.json", manifest)
    print(json.dumps({"gate_s_qat": manifest["status"], "manifest": str(args.result_dir / "manifest.json"), "external_data_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
