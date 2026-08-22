"""Revalidate M0/M1/M2 and L1/L2 under the retained P0/O1 protocol.

Each invocation executes exactly one frozen outer-fold and seed combination.
Only MIT-BIH, INCART, and LTDDB training/development records are allowed.
SVDB and NSRDB are rejected before any record is loaded. Results are written to
a new directory and an existing primary result file is never overwritten.
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
SEEDS = (20260803, 20260804, 20260805)
FORBIDDEN_SOURCES = {"svdb", "nsrdb"}
CONDITIONS = {
    "M0_L2": {"source_condition": "M0", "sources": {"mitdb"}, "lead_profile": "L2", "lead_count": 2},
    "M1_L2": {"source_condition": "M1", "sources": {"mitdb", "incartdb"}, "lead_profile": "L2", "lead_count": 2},
    "M2_L2": {"source_condition": "M2", "sources": {"mitdb", "incartdb", "ltdb"}, "lead_profile": "L2", "lead_count": 2},
    "M1_L1": {"source_condition": "M1", "sources": {"mitdb", "incartdb"}, "lead_profile": "L1", "lead_count": 1},
}


def load_core():
    path = Path(__file__).with_name("14_experiment2_5_optimize.py")
    spec = importlib.util.spec_from_file_location("final_revalidation_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the reviewed Experiment 2.5 core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load_core()


class FinalRecordStore(core.RecordStore):
    """P0/O1 record cache extended only to the reviewed M0/M1/M2 sources."""

    def __init__(self, records: list, data_root: Path, cache_dir: Path, lead_count: int = 2) -> None:
        self.records = records
        self.data_root = data_root
        self.cache_dir = cache_dir
        self.lead_count = lead_count
        self.memory = core.OrderedDict()
        self.cache_size = 8
        self.by_source_class = core.defaultdict(lambda: core.defaultdict(list))
        allowed_sources = {"mitdb", "incartdb", "ltdb"}
        for record in records:
            if record.source not in allowed_sources or record.source in FORBIDDEN_SOURCES:
                raise ValueError(f"Final revalidation source violation: {record.source}")
            if record.normal_count:
                self.by_source_class[record.source][0].append(record)
            if record.pvc_count:
                self.by_source_class[record.source][1].append(record)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_p0_model(lead_count: int):
    """Construct O1/P0 with only the preregistered lead-count ablation varied."""
    import tensorflow as tf

    waveform = tf.keras.Input(shape=(300, lead_count), name="ecg_beat")
    morphology = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu")(waveform)
    morphology = tf.keras.layers.MaxPooling1D(2)(morphology)
    morphology = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(morphology)
    morphology = tf.keras.layers.GlobalAveragePooling1D()(morphology)
    rr_inputs, rr_features = core.rr_branch(tf)
    fused = tf.keras.layers.Concatenate(name="fusion")([morphology, rr_features])
    hidden = tf.keras.layers.Dense(16, activation="relu", name="classifier_hidden")(fused)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="class")(hidden)
    model = tf.keras.Model([waveform, rr_inputs], outputs, name=f"p0_revalidate_l{lead_count}")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def train_epochs(records, store, args, seed: int, epochs: int, lead_count: int):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    sampler = core.HierarchicalSampler(records, store, seed, augment=False)
    model = build_p0_model(lead_count)
    for _ in range(epochs):
        for _ in range(args.steps_per_epoch):
            waveforms, rr, labels = sampler.sample(args.batch_size)
            model.train_on_batch([waveforms, rr], labels)
    return model


def select_epoch(train_records, validation_records, store, args, seed: int, lead_count: int):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    sampler = core.HierarchicalSampler(train_records, store, seed, augment=False)
    model = build_p0_model(lead_count)
    curve, best_epoch, best_score, best_weights = [], 1, -np.inf, None
    for epoch in range(1, args.epochs + 1):
        for _ in range(args.steps_per_epoch):
            waveforms, rr, labels = sampler.sample(args.batch_size)
            model.train_on_batch([waveforms, rr], labels)
        predictions = core.predict_records(model, "O1", validation_records, store)
        _, metrics = core.aggregate_records(predictions, 0.5)
        score = metrics["record_macro"]["record_macro_auprc"]
        curve.append({"epoch": epoch, "selection_record_macro_auprc": score})
        if score is not None and score > best_score:
            best_epoch, best_score, best_weights = epoch, score, model.get_weights()
    if best_weights is None:
        raise RuntimeError("Inner epoch selection did not produce a usable model")
    model.set_weights(best_weights)
    return model, best_epoch, curve


def write_protocol(output_dir: Path) -> None:
    protocol_conditions = {
        name: {
            **configuration,
            "sources": sorted(configuration["sources"]),
        }
        for name, configuration in CONDITIONS.items()
    }
    payload = {
        "experiment": "final_architecture_m0_m1_m2_l1_l2_revalidation",
        "candidate": "P0/O1 retained final protocol",
        "conditions": protocol_conditions,
        "seeds": list(SEEDS),
        "split_sha256": SPLIT_SHA256,
        "forbidden_sources": sorted(FORBIDDEN_SOURCES),
        "external_data_accessed": False,
    }
    path = output_dir / "protocol.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("Existing revalidation protocol does not match this run")
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--manifest", type=Path, default=Path("results/experiment0/record_manifest.csv"))
    parser.add_argument("--splits", type=Path, default=Path("results/experiment1/splits.json"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_revalidation"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 20 <= args.epochs <= 30:
        parser.error("Revalidation retains the preregistered 20-30 epoch budget")
    if args.steps_per_epoch < 1 or args.batch_size < 2:
        parser.error("Steps per epoch and batch size must be positive")
    configuration = CONDITIONS[args.condition]
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    if split_payload.get("split_sha256") != SPLIT_SHA256:
        raise ValueError("Split manifest hash is not the reviewed value")
    source_condition = split_payload["conditions"].get(configuration["source_condition"])
    if source_condition is None or set(source_condition["sources"]) != configuration["sources"]:
        raise ValueError("Selected condition does not match the reviewed source split")
    if not 1 <= args.outer_fold <= len(source_condition["outer_folds"]):
        parser.error("Outer fold is outside the reviewed split manifest")

    records = [
        record
        for record in core.prep.load_training_records(args.manifest)
        if record.source in configuration["sources"]
    ]
    observed_sources = {record.source for record in records}
    if observed_sources != configuration["sources"] or observed_sources & FORBIDDEN_SOURCES:
        raise ValueError("Training record source boundary check failed")
    test_keys = set(source_condition["outer_folds"][args.outer_fold - 1]["records"])
    development = [record for record in records if record.key not in test_keys]
    test = [record for record in records if record.key in test_keys]
    if not development or not test or {record.key for record in development} & {record.key for record in test}:
        raise AssertionError("Outer-fold record isolation check failed")
    if args.dry_run:
        print(json.dumps({
            "status": "passed",
            "condition": args.condition,
            "sources": sorted(observed_sources),
            "lead_profile": configuration["lead_profile"],
            "development_records": len(development),
            "outer_test_records": len(test),
            "external_data_accessed": False,
        }, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_protocol(args.output_dir)
    result_path = args.output_dir / f"{args.condition}_fold{args.outer_fold}_seed{args.seed}.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {result_path}")
    cache_dir = args.output_dir / "window_cache" / args.condition
    store = FinalRecordStore(records, args.data_root, cache_dir, lead_count=configuration["lead_count"])
    inner_folds = [fold for index, fold in enumerate(source_condition["outer_folds"]) if index != args.outer_fold - 1]
    oof, curves, selected_epochs = {}, {}, []
    for inner_index, fold in enumerate(inner_folds):
        validation_keys = set(fold["records"])
        inner_train = [record for record in development if record.key not in validation_keys]
        inner_validation = [record for record in development if record.key in validation_keys]
        model, epoch, curve = select_epoch(
            inner_train,
            inner_validation,
            store,
            args,
            args.seed + inner_index,
            configuration["lead_count"],
        )
        oof.update(core.predict_records(model, "O1", inner_validation, store))
        curves[f"inner_{inner_index + 1}"] = curve
        selected_epochs.append(epoch)
    scaler, threshold, selection = core.calibrate_and_select(oof)
    final_epochs = int(np.median(selected_epochs))
    final_model = train_epochs(
        development,
        store,
        args,
        args.seed + 100,
        final_epochs,
        configuration["lead_count"],
    )
    raw_predictions = core.predict_records(final_model, "O1", test, store)
    calibrated = {
        key: (labels, core.apply_scaler(scaler, probabilities))
        for key, (labels, probabilities) in raw_predictions.items()
    }
    per_record, outer_test = core.aggregate_records(calibrated, threshold)
    prediction_dir = args.output_dir / "per_record_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for key, (labels, probabilities) in calibrated.items():
        prediction_path = prediction_dir / f"{args.condition}_fold{args.outer_fold}_seed{args.seed}_{key.replace(':', '_')}.npz"
        if prediction_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing prediction: {prediction_path}")
        np.savez_compressed(prediction_path, labels=labels, calibrated_probability=probabilities)
    curve_dir = args.output_dir / "learning_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    curve_path = curve_dir / f"{args.condition}_fold{args.outer_fold}_seed{args.seed}.json"
    if curve_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing learning curve: {curve_path}")
    curve_path.write_text(json.dumps(curves, indent=2) + "\n", encoding="utf-8")
    architecture = final_model.to_json()
    result = {
        "experiment": "final_architecture_m0_m1_m2_l1_l2_revalidation",
        "candidate": "P0/O1",
        "condition": args.condition,
        "source_condition": configuration["source_condition"],
        "sources": sorted(observed_sources),
        "lead_profile": configuration["lead_profile"],
        "input_shape": [300, configuration["lead_count"]],
        "outer_fold": args.outer_fold,
        "seed": args.seed,
        "split_sha256": SPLIT_SHA256,
        "external_data_accessed": False,
        "training": {
            "max_epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "final_epochs": final_epochs,
            "inner_selected_epochs": selected_epochs,
            "sampler": "hierarchical source -> record -> class",
            "augmentation": False,
        },
        "selection": selection | {
            "threshold": threshold,
            "epoch_selection_metric": "inner record-macro AUPRC",
            "calibration": "P0 Platt scaling with source- and record-balanced fitting weights",
        },
        "architecture": {
            "parameter_count": int(final_model.count_params()),
            "json_sha256": hashlib.sha256(architecture.encode("utf-8")).hexdigest(),
        },
        "outer_test": outer_test,
        "per_record": per_record,
        "source_code_sha256": sha256_file(Path(__file__)),
        "core_source_code_sha256": sha256_file(Path(__file__).with_name("14_experiment2_5_optimize.py")),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), "condition": args.condition, "external_data_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
