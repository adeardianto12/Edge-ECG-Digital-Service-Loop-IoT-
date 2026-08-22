"""Run a conversion-only Gate S compatibility check for retained P0/O1.

This remediation deliberately reuses the saved float model and its already
frozen Platt calibration/threshold.  It neither trains nor selects a model,
calibration, threshold, representative-data rule, or external dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np


SOURCES = {"mitdb", "incartdb"}
FORBIDDEN_SOURCES = {"nsrdb", "svdb"}
SPLIT_SHA256 = "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787"
EXPECTED_PARAMETER_COUNT = 3506


def load_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("26_gate_s_freeze.py", "gate_s_base")
revalidation = base.revalidation
core = base.core


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def require_new_targets(model_dir: Path, result_dir: Path) -> None:
    for directory in (model_dir, result_dir):
        if directory.exists():
            raise FileExistsError(f"Refusing to reuse conversion target: {directory}")


def load_m1_records(splits_path: Path):
    """Construct M1 records from the frozen M1 split, without reading external rows."""
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    condition = payload.get("conditions", {}).get("M1")
    if payload.get("split_sha256") != SPLIT_SHA256 or not condition:
        raise ValueError("Frozen M1 split is unavailable or has an unexpected hash")
    if set(condition.get("sources", [])) != SOURCES:
        raise ValueError("Frozen M1 split does not contain exactly MIT-BIH and INCART")
    keys = []
    for fold in condition.get("outer_folds", []):
        keys.extend(fold.get("records", []))
    if len(keys) != len(set(keys)):
        raise ValueError("Frozen M1 folds do not partition records exactly once")
    records = []
    for key in keys:
        source, separator, record = key.partition(":")
        if not separator or source not in SOURCES or source in FORBIDDEN_SOURCES:
            raise ValueError(f"Invalid or forbidden record key in M1 split: {key}")
        # Counts and group membership are irrelevant for conversion sampling; the
        # store only needs the immutable source and record identity.
        records.append(core.prep.Record(source, record, key, 0, 0))
    if {record.source for record in records} != SOURCES:
        raise ValueError("M1 records do not cover both development sources")
    return records


def frozen_scaler(calibration: dict):
    """Rebuild the saved Platt object so all probability logic stays in core."""
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


def load_source_model(tf, source_model: Path, extracted_weights_path: Path):
    """Load the frozen Keras 3 model, or reconstruct its declared graph for Keras 2."""
    try:
        return tf.keras.models.load_model(source_model, compile=False), {
            "method": "direct_keras_model_load",
        }
    except (ModuleNotFoundError, TypeError, ValueError) as error:
        # TensorFlow 2.15 bundles Keras 2 and cannot deserialize the Keras 3
        # Functional class.  Its HDF5 weight archive remains portable, so load
        # those exact arrays into the already-declared P0/O1 graph.
        with zipfile.ZipFile(source_model) as archive:
            weights_bytes = archive.read("model.weights.h5")
        extracted_weights_path.write_bytes(weights_bytes)
        import h5py

        model = revalidation.build_p0_model(2)
        layer_map = {
            "conv1d": "conv1d",
            "conv1d_1": "conv1d_1",
            "dense": "rr_mlp",
            "dense_1": "classifier_hidden",
            "dense_2": "class",
        }
        with h5py.File(extracted_weights_path, "r") as handle:
            for source_layer, target_layer in layer_map.items():
                variables = handle["layers"][source_layer]["vars"]
                values = [np.asarray(variables[str(index)]) for index in range(len(variables))]
                layer = model.get_layer(target_layer)
                if [tuple(value.shape) for value in values] != [tuple(weight.shape) for weight in layer.get_weights()]:
                    raise ValueError(f"Frozen weight shape mismatch for {source_layer}")
                layer.set_weights(values)
        return model, {
            "method": "declared_p0_o1_graph_with_exact_weight_archive",
            "direct_load_error": f"{type(error).__name__}: {error}",
            "weight_archive": str(extracted_weights_path),
            "weight_archive_sha256": hashlib.sha256(weights_bytes).hexdigest(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("results/experiment1/splits.json"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--source-model-dir", type=Path, default=Path("models/software_freeze_gate_s_retry6"))
    parser.add_argument("--source-result-dir", type=Path, default=Path("results/software_freeze_gate_s_retry6"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--toolchain-label", required=True)
    parser.add_argument("--quantization-samples", type=int, default=5000)
    parser.add_argument("--equivalence-samples", type=int, default=1000)
    args = parser.parse_args()

    if args.quantization_samples < 1 or args.equivalence_samples < 1:
        parser.error("Sample counts must be positive")
    require_new_targets(args.model_dir, args.result_dir)
    source_model = args.source_model_dir / "model.keras"
    source_calibration = args.source_model_dir / "calibration.json"
    source_manifest = args.source_result_dir / "manifest.json"
    for path in (source_model, source_calibration, source_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Required source artifact is missing: {path}")

    calibration = json.loads(source_calibration.read_text(encoding="utf-8"))
    if calibration.get("threshold") is None:
        raise ValueError("Frozen threshold is missing")
    records = load_m1_records(args.splits)

    args.model_dir.mkdir(parents=True, exist_ok=False)
    args.result_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "gate": "Gate S: Software Evaluation Freeze",
        "remediation": "conversion-toolchain compatibility check only",
        "candidate": "P0/O1",
        "source_float_model": str(source_model),
        "source_float_model_sha256": sha256_file(source_model),
        "source_calibration_sha256": sha256_file(source_calibration),
        "source_failed_attempt_manifest_sha256": sha256_file(source_manifest),
        "sources": sorted(SOURCES),
        "forbidden_sources": sorted(FORBIDDEN_SOURCES),
        "external_data_accessed": False,
        "split_sha256": SPLIT_SHA256,
        "calibration_and_threshold": "reused unchanged from source float artifact; no int8 or external-data selection",
        "representative_data": "M1 development only; fixed record-balanced selection from existing Gate S rule",
        "acceptance": {
            "int8_size_bytes_lt": 1048576,
            "decision_agreement_gte": 0.99,
            "mean_calibrated_probability_error_lte": 0.02,
            "p99_calibrated_probability_error_lte": 0.05,
            "support_aware_f1_loss_lte": 0.01,
            "recall_loss_lte": 0.01,
        },
        "overwrite_policy": "new model and result directories only",
        "toolchain_label": args.toolchain_label,
    }
    write_json(args.result_dir / "protocol.json", protocol)

    import tensorflow as tf

    model, source_load = load_source_model(tf, source_model, args.result_dir / "source_model.weights.h5")
    if int(model.count_params()) != EXPECTED_PARAMETER_COUNT:
        raise AssertionError("Source float model does not match retained P0/O1")
    cache_dir = args.result_dir / "window_cache"
    store = revalidation.FinalRecordStore(records, args.data_root, cache_dir, lead_count=2)
    representative_waveforms, representative_rr, held_waveforms, held_rr, held_labels = base.select_disjoint_samples(
        records, store, args.quantization_samples, args.equivalence_samples
    )
    copied_model = args.model_dir / "model.keras"
    copied_calibration = args.model_dir / "calibration.json"
    shutil.copy2(source_model, copied_model)
    shutil.copy2(source_calibration, copied_calibration)
    tflite_path = args.model_dir / "model_int8.tflite"
    export_model = base.export_int8(model, representative_waveforms, representative_rr, tflite_path)
    float_raw = model.predict([held_waveforms, held_rr], batch_size=128, verbose=0)[:, 1]
    int8_raw = base.tflite_predict(tflite_path, held_waveforms, held_rr)
    threshold = float(calibration["threshold"])
    scaler = frozen_scaler(calibration)
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
        "experiment": "gate_s_converter_compatibility_int8_equivalence",
        "candidate": "P0/O1 retained source float model",
        "toolchain_label": args.toolchain_label,
        "equivalence_set": {
            "source": "M1 development only; held disjoint from representative quantization samples",
            "samples": int(len(held_labels)),
            "quantization_samples": int(len(representative_waveforms)),
            "sampling_seed": 20260811,
        },
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
    manifest = {
        "gate": "Gate S: Software Evaluation Freeze",
        "status": "passed" if equivalence["passed"] else "failed",
        "remediation": "conversion-toolchain compatibility check only; no retraining or model selection",
        "candidate": "P0/O1",
        "source_condition": "M1: MIT-BIH + INCART",
        "lead_profile": "L2",
        "parameter_count": int(model.count_params()),
        "input_contract": {"waveform": [300, 2], "causal_rr": [5]},
        "calibration": {key: calibration[key] for key in ("method", "coefficient", "intercept", "threshold")},
        "source_artifacts": {
            "float_model": str(source_model),
            "float_model_sha256": sha256_file(source_model),
            "calibration_sha256": sha256_file(source_calibration),
            "prior_attempt_manifest_sha256": sha256_file(source_manifest),
            "load_path": source_load,
        },
        "artifacts": {
            "keras": str(copied_model),
            "keras_sha256": sha256_file(copied_model),
            "int8_tflite": str(tflite_path),
            "int8_tflite_sha256": sha256_file(tflite_path),
            "export_architecture_json_sha256": hashlib.sha256(export_model.to_json().encode("utf-8")).hexdigest(),
        },
        "dependency_versions": dependency_versions(),
        "split_sha256": SPLIT_SHA256,
        "external_data_accessed": False,
        "prohibited_sources": sorted(FORBIDDEN_SOURCES),
        "int8_equivalence": {"path": str(args.result_dir / "int8_equivalence.json"), "sha256": sha256_file(args.result_dir / "int8_equivalence.json"), "passed": equivalence["passed"]},
        "immutable": True,
    }
    write_json(args.result_dir / "manifest.json", manifest)
    print(json.dumps({"gate_s": manifest["status"], "manifest": str(args.result_dir / "manifest.json"), "external_data_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
