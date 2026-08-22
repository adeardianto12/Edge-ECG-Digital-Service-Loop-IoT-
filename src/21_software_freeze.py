"""Create a Gate S software-freeze artifact from a selected Experiment 2.6 candidate.

The command is intentionally blocked until Experiment 2.6 has a complete,
accepted selection.  It fits calibration and thresholds from development-only
OOF predictions, exports one float model and one full-integer int8 model, and
records float/int8 equivalence evidence.  It does not access SVDB or NSRDB.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SPLIT_SHA256 = "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787"
SOURCES = {"mitdb", "incartdb"}
FORBIDDEN_SOURCES = {"svdb", "nsrdb"}


def load_experiment_module():
    path = Path(__file__).with_name("19_experiment2_6_optimize.py")
    spec = importlib.util.spec_from_file_location("experiment2_6_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Experiment 2.6 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


experiment = load_experiment_module()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_candidate(summary_path: Path) -> str:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment") != "experiment_2_6_controlled_refinement":
        raise ValueError("Gate S requires an Experiment 2.6 summary")
    candidate = summary.get("decision", {}).get("selected_candidate")
    if summary.get("decision", {}).get("status") != "selected" or candidate not in experiment.CANDIDATES:
        raise ValueError("Gate S is blocked until Experiment 2.6 selects or explicitly retains a candidate")
    return candidate


def selected_epochs(output_dir: Path, candidate: str) -> int:
    values = []
    for fold in range(1, 6):
        path = output_dir / f"{candidate}_fold{fold}_seed20260803.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("split_sha256") != SPLIT_SHA256 or payload.get("external_data_accessed"):
            raise ValueError(f"Invalid selected-candidate evidence: {path}")
        values.append(int(payload["training"]["final_epochs"]))
    return int(np.median(values))


def representative_data(records, store, limit: int = 1000):
    rng = np.random.default_rng(20260809)
    waveforms, rr = [], []
    for record in records:
        loaded = store.load(record)
        if not len(loaded["labels"]):
            continue
        positions = rng.choice(len(loaded["labels"]), size=min(16, len(loaded["labels"])), replace=False)
        waveforms.extend(loaded["waveforms"][position] for position in positions)
        rr.extend(loaded["rr"][position] for position in positions)
        if len(waveforms) >= limit:
            break
    if not waveforms:
        raise ValueError("No development representative samples are available")
    labels = []
    for record in records:
        loaded = store.load(record)
        if len(loaded["labels"]):
            labels.extend(loaded["labels"][:min(16, len(loaded["labels"]))])
            if len(labels) >= limit:
                break
    return np.asarray(waveforms[:limit], dtype=np.float32), np.asarray(rr[:limit], dtype=np.float32), np.asarray(labels[:limit], dtype=np.int32)


def quantize(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    if scale == 0:
        return values.astype(detail["dtype"])
    info = np.iinfo(detail["dtype"])
    return np.clip(np.rint(values / scale + zero_point), info.min, info.max).astype(detail["dtype"])


def dequantize(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    return values.astype(np.float32) * scale + zero_point


def tflite_predict(model_path: Path, waveforms: np.ndarray, rr: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 2 or len(outputs) != 1:
        raise ValueError("Gate S expects exactly two model inputs and one class-probability output")
    waveform_input = next((detail for detail in inputs if tuple(detail["shape"][1:]) == (300, 2)), None)
    rr_input = next((detail for detail in inputs if tuple(detail["shape"][1:]) == (5,)), None)
    if waveform_input is None or rr_input is None:
        raise ValueError("Unexpected TFLite input contract")
    output = []
    for start in range(0, len(waveforms), 128):
        end = min(start + 128, len(waveforms))
        interpreter.resize_tensor_input(waveform_input["index"], [end - start, 300, 2], strict=False)
        interpreter.resize_tensor_input(rr_input["index"], [end - start, 5], strict=False)
        interpreter.allocate_tensors()
        inputs = interpreter.get_input_details()
        waveform_input = next(detail for detail in inputs if tuple(detail["shape"][1:]) == (300, 2))
        rr_input = next(detail for detail in inputs if tuple(detail["shape"][1:]) == (5,))
        interpreter.set_tensor(waveform_input["index"], quantize(waveforms[start:end], waveform_input))
        interpreter.set_tensor(rr_input["index"], quantize(rr[start:end], rr_input))
        interpreter.invoke()
        output_detail = interpreter.get_output_details()[0]
        output.append(dequantize(interpreter.get_tensor(output_detail["index"]), output_detail))
    return np.concatenate(output, axis=0)[:, 1]


def export_int8(model, representative_waveforms: np.ndarray, representative_rr: np.ndarray, output_path: Path) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def generator():
        for index in range(len(representative_waveforms)):
            yield [representative_waveforms[index : index + 1], representative_rr[index : index + 1]]

    converter.representative_dataset = generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output_path.write_bytes(converter.convert())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path("results/experiment2_6"))
    parser.add_argument("--manifest", type=Path, default=Path("results/experiment0/record_manifest.csv"))
    parser.add_argument("--splits", type=Path, default=Path("results/experiment1/splits.json"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--model-dir", type=Path, default=Path("models/software_freeze"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/software_freeze"))
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    candidate = selected_candidate(args.experiment_dir / "summary.json")
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    if split_payload.get("split_sha256") != SPLIT_SHA256:
        raise ValueError("Gate S cannot use an unreviewed split manifest")
    condition = split_payload["conditions"].get("M1")
    if condition is None or set(condition["sources"]) != SOURCES:
        raise ValueError("Gate S is restricted to MIT-BIH plus INCART")
    records = [record for record in experiment.core.prep.load_training_records(args.manifest) if record.source in SOURCES]
    if {record.source for record in records} != SOURCES:
        raise ValueError("M1 development records are incomplete")
    epochs = selected_epochs(args.experiment_dir, candidate)
    store_class = experiment.RobustRRStore if candidate == "P2" else experiment.core.RecordStore
    store = store_class(records, args.data_root, args.experiment_dir / "window_cache")
    oof = {}
    for fold_index, fold in enumerate(condition["outer_folds"], start=1):
        test_keys = set(fold["records"])
        train = [record for record in records if record.key not in test_keys]
        test = [record for record in records if record.key in test_keys]
        model = experiment.train_epochs(candidate, train, store, args, 20260803 + fold_index, epochs)
        oof.update(experiment.core.predict_records(model, "O1", test, store))
    scaler, threshold, selection = experiment.calibrate_and_select(oof, candidate)
    final_model = experiment.train_epochs(candidate, records, store, args, 20260803, epochs)
    representative_waveforms, representative_rr, representative_labels = representative_data(records, store)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    keras_path = args.model_dir / "model.keras"
    tflite_path = args.model_dir / "model_int8.tflite"
    final_model.save(keras_path)
    export_int8(final_model, representative_waveforms, representative_rr, tflite_path)
    float_raw = final_model.predict([representative_waveforms, representative_rr], batch_size=128, verbose=0)[:, 1]
    int8_raw = tflite_predict(tflite_path, representative_waveforms, representative_rr)
    float_probability = experiment.apply_scaler(scaler, float_raw, candidate)
    int8_probability = experiment.apply_scaler(scaler, int8_raw, candidate)
    errors = np.abs(float_probability - int8_probability)
    decision_agreement = float(np.mean((float_probability >= threshold) == (int8_probability >= threshold)))
    def metrics(probability):
        decision = probability >= threshold
        true_positive = int(np.sum(decision & (representative_labels == 1)))
        false_positive = int(np.sum(decision & (representative_labels == 0)))
        false_negative = int(np.sum(~decision & (representative_labels == 1)))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        return {"f1": 2 * precision * recall / max(precision + recall, 1e-12), "recall": recall}
    float_metrics, int8_metrics = metrics(float_probability), metrics(int8_probability)
    equivalence = {
        "experiment": "software_freeze_int8_equivalence",
        "candidate": candidate,
        "representative_samples": int(len(representative_waveforms)),
        "float_int8_decision_agreement": decision_agreement,
        "mean_absolute_calibrated_probability_error": float(np.mean(errors)),
        "p99_absolute_calibrated_probability_error": float(np.quantile(errors, 0.99)),
        "int8_size_bytes": tflite_path.stat().st_size,
        "criteria": {"size_under_1_mib": tflite_path.stat().st_size < 1024 * 1024, "decision_agreement_at_least_0_99": decision_agreement >= 0.99, "mean_error_at_most_0_02": float(np.mean(errors)) <= 0.02, "p99_error_at_most_0_05": float(np.quantile(errors, 0.99)) <= 0.05, "f1_loss_at_most_0_01": float_metrics["f1"] - int8_metrics["f1"] <= 0.01, "recall_loss_at_most_0_01": float_metrics["recall"] - int8_metrics["recall"] <= 0.01},
        "float_metrics": float_metrics,
        "int8_metrics": int8_metrics,
    }
    equivalence["passed"] = all(equivalence["criteria"].values())
    (args.result_dir / "int8_equivalence.json").write_text(json.dumps(equivalence, indent=2) + "\n", encoding="utf-8")
    calibration = {"method": selection["calibration"], "coefficient": float(scaler.coef_[0][0]), "intercept": float(scaler.intercept_[0]), "threshold": threshold}
    (args.model_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "gate": "Gate S: Software Evaluation Freeze",
        "status": "passed" if equivalence["passed"] else "failed",
        "candidate": candidate,
        "seed": 20260803,
        "source_condition": "M1: MIT-BIH + INCART",
        "lead_profile": "L2",
        "input_contract": {"waveform": [300, 2], "causal_rr": [5]},
        "epochs": epochs,
        "split_sha256": SPLIT_SHA256,
        "calibration": calibration,
        "artifacts": {"keras": str(keras_path), "keras_sha256": sha256_file(keras_path), "int8_tflite": str(tflite_path), "int8_tflite_sha256": sha256_file(tflite_path), "source_code_sha256": sha256_file(Path(__file__))},
        "external_data_accessed": False,
        "prohibited_sources": sorted(FORBIDDEN_SOURCES),
        "pending_deployment_evidence": ["declared_gateway_latency", "gateway_memory", "endpoint_transport", "energy", "Gate H"],
    }
    (args.result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate_s": manifest["status"], "candidate": candidate, "manifest": str(args.result_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
