"""Execute the single registered standard-TFMOT Gate S QAT remediation."""

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


def load_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compat = load_module("29_gate_s_converter_compatibility.py", "gate_s_compat")
base = compat.base
revalidation = compat.revalidation
core = compat.core


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite QAT artifact: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_registered_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "registered":
        raise ValueError("Standard QAT protocol is not registered")
    if protocol.get("split_sha256") != SPLIT_SHA256 or set(protocol.get("data_sources", [])) != SOURCES:
        raise ValueError("Standard QAT protocol source boundary is invalid")
    return protocol


def load_records_from_m1_split(splits_path: Path, data_root: Path) -> list:
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    condition = payload.get("conditions", {}).get("M1")
    if payload.get("split_sha256") != SPLIT_SHA256 or not condition:
        raise ValueError("Frozen M1 split is unavailable or has an unexpected hash")
    keys = [key for fold in condition.get("outer_folds", []) for key in fold.get("records", [])]
    if len(keys) != len(set(keys)) or set(condition.get("sources", [])) != SOURCES:
        raise ValueError("Frozen M1 split is not a disjoint MIT-BIH + INCART partition")
    from multisource_ecg import SOURCE_BY_KEY, load_pvc_windows

    records = []
    for key in keys:
        source, separator, record_name = key.partition(":")
        if not separator or source not in SOURCES or source in FORBIDDEN_SOURCES:
            raise ValueError(f"Invalid or forbidden M1 record: {key}")
        spec = SOURCE_BY_KEY[source]
        loaded = load_pvc_windows(data_root / spec.directory / record_name, spec, lead_count=2)
        labels = loaded["labels"]
        records.append(core.prep.Record(source, record_name, key, int(np.sum(labels == 0)), int(np.sum(labels == 1))))
    if {record.source for record in records} != SOURCES:
        raise ValueError("M1 development records do not cover both sources")
    return records


def make_conv1d_config():
    import tensorflow_model_optimization as tfmot

    quantizers = tfmot.quantization.keras.quantizers

    class Conv1DConfig(tfmot.quantization.keras.QuantizeConfig):
        def get_weights_and_quantizers(self, layer):
            # TFMOT 0.8's per-axis kernel reducer assumes Conv2D rank; Conv1D
            # uses the same standard 8-bit scheme with a per-tensor kernel range.
            return [(layer.kernel, quantizers.LastValueQuantizer(8, False, True, True))]

        def get_activations_and_quantizers(self, layer):
            return []

        def set_quantize_weights(self, layer, quantized_weights):
            layer.kernel.assign(quantized_weights[0])

        def set_quantize_activations(self, layer, quantized_activations):
            del quantized_activations

        def get_output_quantizers(self, layer):
            del layer
            return [quantizers.MovingAverageQuantizer(8, False, False, False)]

        def get_config(self):
            return {}

    return Conv1DConfig


def build_plain_model(batch_one: bool = False):
    import tf_keras as keras

    waveform = keras.Input(batch_shape=(1, 300, 2) if batch_one else None, shape=None if batch_one else (300, 2), name="ecg_beat")
    morphology = keras.layers.Conv1D(16, 5, padding="same", activation="relu", name="conv1d")(waveform)
    morphology = keras.layers.MaxPooling1D(2, name="max_pooling1d")(morphology)
    morphology = keras.layers.Conv1D(32, 5, padding="same", activation="relu", name="conv1d_1")(morphology)
    morphology = keras.layers.GlobalAveragePooling1D(name="global_average_pooling1d")(morphology)
    rr_inputs = keras.Input(batch_shape=(1, 5) if batch_one else None, shape=None if batch_one else (5,), name="causal_rr")
    rr_features = keras.layers.Dense(8, activation="relu", name="rr_mlp")(rr_inputs)
    fused = keras.layers.Concatenate(name="fusion")([morphology, rr_features])
    hidden = keras.layers.Dense(16, activation="relu", name="classifier_hidden")(fused)
    outputs = keras.layers.Dense(2, activation="softmax", name="class")(hidden)
    return keras.Model([waveform, rr_inputs], outputs, name="p0_o1_plain_export")


def build_qat_model(batch_one: bool = False):
    import tensorflow as tf
    import tf_keras as keras
    import tensorflow_model_optimization as tfmot

    Conv1DConfig = make_conv1d_config()
    waveform = keras.Input(batch_shape=(1, 300, 2) if batch_one else None, shape=None if batch_one else (300, 2), name="ecg_beat")
    morphology = keras.layers.Conv1D(16, 5, padding="same", activation="relu")(waveform)
    morphology = keras.layers.MaxPooling1D(2)(morphology)
    morphology = keras.layers.Conv1D(32, 5, padding="same", activation="relu")(morphology)
    morphology = keras.layers.GlobalAveragePooling1D()(morphology)
    rr_inputs = keras.Input(batch_shape=(1, 5) if batch_one else None, shape=None if batch_one else (5,), name="causal_rr")
    rr_features = keras.layers.Dense(8, activation="relu", name="rr_mlp")(rr_inputs)
    fused = keras.layers.Concatenate(name="fusion")([morphology, rr_features])
    hidden = keras.layers.Dense(16, activation="relu", name="classifier_hidden")(fused)
    outputs = keras.layers.Dense(2, activation="softmax", name="class")(hidden)
    # TFMOT's direct wrapper path avoids Keras-module serialization ambiguity
    # while retaining the standard QAT fake-quantization implementation.
    def wrap(layer):
        return tfmot.quantization.keras.QuantizeWrapper(layer, Conv1DConfig(), name=layer.name)

    waveform = keras.Input(shape=(300, 2), name="ecg_beat")
    morphology = wrap(keras.layers.Conv1D(16, 5, padding="same", activation="relu", name="conv1d"))(waveform)
    morphology = keras.layers.MaxPooling1D(2, name="max_pooling1d")(morphology)
    morphology = wrap(keras.layers.Conv1D(32, 5, padding="same", activation="relu", name="conv1d_1"))(morphology)
    morphology = keras.layers.GlobalAveragePooling1D(name="global_average_pooling1d")(morphology)
    rr_inputs = keras.Input(shape=(5,), name="causal_rr")
    rr_features = wrap(keras.layers.Dense(8, activation="relu", name="rr_mlp"))(rr_inputs)
    fused = keras.layers.Concatenate(name="fusion")([morphology, rr_features])
    hidden = wrap(keras.layers.Dense(16, activation="relu", name="classifier_hidden"))(fused)
    outputs = wrap(keras.layers.Dense(2, activation="softmax", name="class"))(hidden)
    qat_model = keras.Model([waveform, rr_inputs], outputs, name="p0_o1_standard_qat")
    trainable_count = sum(int(np.prod(weight.shape)) for weight in qat_model.trainable_weights)
    if trainable_count != PARAMETERS:
        raise AssertionError(f"QAT trainable parameter count changed: {trainable_count}")
    return qat_model


def train_qat(records, store, epochs: int, steps_per_epoch: int, batch_size: int, seed: int):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_qat_model()
    sampler = core.HierarchicalSampler(records, store, seed, augment=False)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            waveforms, rr, labels = sampler.sample(batch_size)
            model.train_on_batch([waveforms, rr], labels)
    return model


def export_int8(qat_model, representative_waveforms, representative_rr, path: Path):
    import tensorflow as tf
    import tf_keras as keras
    import tensorflow_model_optimization as tfmot

    fixed = build_plain_model(batch_one=True)
    for layer in fixed.layers:
        if not layer.weights:
            continue
        wrapper = qat_model.get_layer(layer.name)
        if not hasattr(wrapper, "layer"):
            raise AssertionError(f"Expected QAT wrapper for trainable layer {layer.name}")
        layer.set_weights(wrapper.layer.get_weights())
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
    import tensorflow as tf
    import tensorflow_model_optimization as tfmot
    import wfdb
    import sklearn

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "tensorflow_model_optimization": tfmot.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "wfdb": wfdb.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("results/gate_s_tfmot_qat/protocol.json"))
    parser.add_argument("--splits", type=Path, default=Path("results/experiment1/splits.json"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--model-dir", type=Path, default=Path("models/gate_s_tfmot_qat"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/gate_s_tfmot_qat"))
    args = parser.parse_args()
    if args.model_dir.exists() or args.result_dir.exists():
        raise FileExistsError("Standard QAT output target already exists")
    protocol = load_registered_protocol(args.protocol)
    records = load_records_from_m1_split(args.splits, args.data_root)
    args.model_dir.mkdir(parents=True, exist_ok=False)
    args.result_dir.mkdir(parents=True, exist_ok=False)
    store = revalidation.FinalRecordStore(records, args.data_root, args.result_dir / "window_cache", lead_count=2)

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    condition = split_payload["conditions"]["M1"]
    oof = {}
    for fold_index, fold in enumerate(condition["outer_folds"], start=1):
        test_keys = set(fold["records"])
        train = [record for record in records if record.key not in test_keys]
        test = [record for record in records if record.key in test_keys]
        model = train_qat(train, store, protocol["epochs"], protocol["steps_per_epoch"], protocol["batch_size"], SEED)
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

    final_model = train_qat(records, store, protocol["epochs"], protocol["steps_per_epoch"], protocol["batch_size"], SEED)
    representative_waveforms, representative_rr, held_waveforms, held_rr, held_labels = base.select_disjoint_samples(records, store, 5000, 1000)
    qat_path = args.model_dir / "model_qat.keras"
    tflite_path = args.model_dir / "model_qat_int8.tflite"
    final_model.save(qat_path)
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
        "support_aware_f1_loss_at_most_0_01": float_metrics["pvc_f1"] - int8_metrics["pvc_f1"] <= 0.01,
        "recall_loss_at_most_0_01": float_metrics["pvc_recall"] - int8_metrics["pvc_recall"] <= 0.01,
    }
    equivalence = {
        "experiment": "gate_s_standard_tfmot_qat_int8_equivalence",
        "candidate": "P0/O1 with one registered standard TFMOT QAT remediation",
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
        "remediation": "standard TensorFlow Model Optimization QAT; P0/O1 topology unchanged",
        "candidate": "P0/O1",
        "source_condition": "M1: MIT-BIH + INCART",
        "lead_profile": "L2",
        "parameter_count": PARAMETERS,
        "qat_total_variable_count": int(final_model.count_params()),
        "trainable_parameter_count": sum(int(np.prod(weight.shape)) for weight in final_model.trainable_weights),
        "seed": SEED,
        "epochs": protocol["epochs"],
        "steps_per_epoch": protocol["steps_per_epoch"],
        "batch_size": protocol["batch_size"],
        "split_sha256": SPLIT_SHA256,
        "calibration": {key: calibration[key] for key in ("method", "coefficient", "intercept", "threshold")},
        "artifacts": {"qat_keras": str(qat_path), "qat_keras_sha256": sha256_file(qat_path), "int8_tflite": str(tflite_path), "int8_tflite_sha256": sha256_file(tflite_path), "export_architecture_json_sha256": hashlib.sha256(export_model.to_json().encode("utf-8")).hexdigest()},
        "dependency_versions": dependency_versions(),
        "external_data_accessed": False,
        "prohibited_sources": sorted(FORBIDDEN_SOURCES),
        "registered_protocol_sha256": sha256_file(args.protocol),
        "int8_equivalence": {"path": str(args.result_dir / "int8_equivalence.json"), "sha256": sha256_file(args.result_dir / "int8_equivalence.json"), "passed": equivalence["passed"]},
        "immutable": True,
    }
    write_json(args.result_dir / "manifest.json", manifest)
    print(json.dumps({"gate_s": manifest["status"], "manifest": str(args.result_dir / "manifest.json"), "external_data_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
