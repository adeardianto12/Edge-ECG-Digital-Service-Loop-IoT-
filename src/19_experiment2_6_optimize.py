"""Run one bounded Experiment 2.6 M1/L2 outer-fold candidate.

This runner reuses the reviewed Experiment 2.5 protocol but writes entirely
separate evidence.  It never loads, enumerates, or derives decisions from
SVDB or NSRDB.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from sklearn.linear_model import LogisticRegression

from multisource_ecg import SOURCE_BY_KEY, TARGET_SAMPLE_RATE_HZ, load_pvc_windows


SPLIT_SHA256 = "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787"
SOURCES = {"mitdb", "incartdb"}
FORBIDDEN_SOURCES = {"svdb", "nsrdb"}
CANDIDATES = ("P0", "P1", "P2", "P3")


def load_base_module():
    path = Path(__file__).with_name("14_experiment2_5_optimize.py")
    spec = importlib.util.spec_from_file_location("experiment2_5_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Experiment 2.5 protocol module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load_base_module()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RobustRRStore(core.RecordStore):
    """Causal RR features with explicit clipping for short or abnormal history."""

    def _rr_features(self, record) -> np.ndarray:
        record_base = self.data_root / SOURCE_BY_KEY[record.source].directory / record.record
        header = wfdb.rdheader(str(record_base))
        annotation = wfdb.rdann(str(record_base), "atr")
        peaks = np.unique(np.rint(annotation.sample * TARGET_SAMPLE_RATE_HZ / header.fs).astype(np.int64))
        accepted = load_pvc_windows(record_base, SOURCE_BY_KEY[record.source], lead_count=self.lead_count)
        features = []
        for peak in accepted["canonical_r_peak_samples"]:
            index = int(np.searchsorted(peaks, peak, side="left"))
            preceding = peaks[:index]
            if len(preceding):
                pre_rr = float(peak - preceding[-1]) / TARGET_SAMPLE_RATE_HZ
                history = np.diff(preceding[-9:]) / TARGET_SAMPLE_RATE_HZ
                previous = float(history[-1]) if len(history) else pre_rr
                median = float(np.median(history)) if len(history) else pre_rr
                valid_pre, valid_history = 1.0, float(len(history) >= 8)
            else:
                pre_rr, previous, median, valid_pre, valid_history = 1.0, 1.0, 1.0, 0.0, 0.0
            pre_rr = float(np.clip(pre_rr, 0.3, 2.5))
            previous = float(np.clip(previous, 0.3, 2.5))
            median = float(np.clip(median, 0.3, 2.5))
            features.append([
                pre_rr,
                float(np.clip(pre_rr / median, 0.4, 2.5)),
                float(np.clip(pre_rr / previous, 0.4, 2.5)),
                valid_pre,
                valid_history,
            ])
        return np.asarray(features, dtype=np.float32)

    def load(self, record) -> dict:
        if record.key in self.memory:
            self.memory.move_to_end(record.key)
            return self.memory[record.key]
        directory = self.cache_dir / record.source
        waveform_path = directory / f"{record.record}_waveforms.npy"
        label_path = directory / f"{record.record}_labels.npy"
        rr_path = directory / f"{record.record}_causal_rr_p2_robust.npy"
        if waveform_path.exists() and label_path.exists() and rr_path.exists():
            loaded = {"waveforms": np.load(waveform_path, mmap_mode="r"), "labels": np.load(label_path, mmap_mode="r"), "rr": np.load(rr_path, mmap_mode="r")}
        else:
            directory.mkdir(parents=True, exist_ok=True)
            raw = load_pvc_windows(self.data_root / SOURCE_BY_KEY[record.source].directory / record.record, SOURCE_BY_KEY[record.source], lead_count=self.lead_count)
            loaded = {"waveforms": raw["waveforms"], "labels": raw["labels"], "rr": self._rr_features(record)}
            np.save(waveform_path, loaded["waveforms"])
            np.save(label_path, loaded["labels"])
            np.save(rr_path, loaded["rr"])
        if len(loaded["labels"]) != len(loaded["rr"]):
            raise ValueError(f"Corrupt Experiment 2.6 P2 cache for {record.key}")
        self.memory[record.key] = loaded
        while len(self.memory) > self.cache_size:
            self.memory.popitem(last=False)
        return loaded


def build_model(candidate: str):
    """P0--P2 retain O1; P3 adds only declared RR regularization."""
    if candidate != "P3":
        return core.build_model("O1")
    import tensorflow as tf

    original_branch = core.rr_branch

    def regularized_rr_branch(tf_module):
        inputs = tf_module.keras.Input(shape=(5,), name="causal_rr")
        output = tf_module.keras.layers.Dense(
            8,
            activation="relu",
            kernel_regularizer=tf_module.keras.regularizers.l2(1e-5),
            name="rr_mlp",
        )(inputs)
        return inputs, output

    core.rr_branch = regularized_rr_branch
    try:
        model = core.build_model("O1")
    finally:
        core.rr_branch = original_branch
    return model


def calibrate_and_select(predictions: dict[str, tuple[np.ndarray, np.ndarray]], candidate: str):
    """Fit development-only Platt scaling and one global, recall-safe threshold."""
    labels = np.concatenate([values[0] for values in predictions.values()])
    raw = np.concatenate([values[1] for values in predictions.values()])
    logits = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
    if candidate == "P1":
        logits = np.clip(logits, -5.0, 5.0)
    scaler = LogisticRegression(random_state=0, max_iter=1000).fit(
        logits.reshape(-1, 1), labels, sample_weight=core.balanced_sample_weights(predictions)
    )
    transformed = scaler.predict_proba(logits.reshape(-1, 1))[:, 1]
    calibrated, cursor = {}, 0
    for key, (record_labels, _) in predictions.items():
        calibrated[key] = (record_labels, transformed[cursor : cursor + len(record_labels)])
        cursor += len(record_labels)
    step = 0.02 if candidate == "P1" else 0.01
    options = []
    for threshold in np.arange(0.01, 1.00, step):
        _, summary = core.aggregate_records(calibrated, float(threshold))
        support = summary["record_macro"]["support_aware"]
        options.append({
            "threshold": float(threshold),
            "pooled_recall": summary["pooled"]["pvc_recall"],
            "support_aware_recall": support["pvc_recall"],
            "support_aware_f1": support["pvc_f1"],
            "pvc_free_false_decisions": summary["record_macro"]["pvc_free_records"]["false_pvc_decisions"],
        })
    feasible = [
        item for item in options
        if item["pooled_recall"] >= 0.90 and item["support_aware_recall"] >= 0.90
    ]
    selected = dict(max(
        feasible or options,
        key=lambda item: (
            item["support_aware_f1"],
            item["pooled_recall"],
            -item["pvc_free_false_decisions"],
            item["threshold"],
        ),
    ))
    selected["recall_target_met"] = bool(feasible)
    selected["calibration"] = (
        "robust clipped-logit Platt scaling with a 0.02 threshold grid"
        if candidate == "P1"
        else "Platt scaling with source- and record-balanced fitting weights"
    )
    selected["threshold_trace"] = options
    return scaler, float(selected["threshold"]), selected


def apply_scaler(scaler: LogisticRegression, probabilities: np.ndarray, candidate: str) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))
    if candidate == "P1":
        logits = np.clip(logits, -5.0, 5.0)
    return scaler.predict_proba(logits.reshape(-1, 1))[:, 1]


def train_epochs(candidate, records, store, args, seed, epochs):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    sampler = core.HierarchicalSampler(records, store, seed, augment=False)
    model = build_model(candidate)
    for _ in range(epochs):
        for _ in range(args.steps_per_epoch):
            waveforms, rr, labels = sampler.sample(args.batch_size)
            weights = np.where(labels == 1, 1.10, 1.0).astype(np.float32) if candidate == "P3" else None
            model.train_on_batch(core.model_inputs("O1", waveforms, rr), labels, sample_weight=weights)
    return model


def train_select_epoch(candidate, train_records, validation_records, store, args, seed):
    model = train_epochs(candidate, train_records, store, args, seed, epochs=0)
    sampler = core.HierarchicalSampler(train_records, store, seed, augment=False)
    curve, best_epoch, best_score, best_weights = [], 1, -np.inf, None
    for epoch in range(1, args.epochs + 1):
        for _ in range(args.steps_per_epoch):
            waveforms, rr, labels = sampler.sample(args.batch_size)
            weights = np.where(labels == 1, 1.10, 1.0).astype(np.float32) if candidate == "P3" else None
            model.train_on_batch(core.model_inputs("O1", waveforms, rr), labels, sample_weight=weights)
        predictions = core.predict_records(model, "O1", validation_records, store)
        _, metrics = core.aggregate_records(predictions, 0.5)
        score = metrics["record_macro"]["record_macro_auprc"]
        curve.append({"epoch": epoch, "selection_record_macro_auprc": score})
        if score is not None and score > best_score:
            best_epoch, best_score, best_weights = epoch, score, model.get_weights()
    if best_weights is None:
        raise RuntimeError("Inner epoch selection did not produce a valid model")
    model.set_weights(best_weights)
    return model, best_epoch, curve


def write_protocol(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "experiment_2_6_controlled_refinement",
        "candidates": {
            "P0": "O1 exact reproduction control",
            "P1": "O1 plus robust clipped-logit Platt calibration and threshold grid",
            "P2": "O1 plus causal RR clipping and validity-aware normalization",
            "P3": "O1 plus mild positive-class loss weighting and RR-branch L2 regularization",
        },
        "stage_a_seed": 20260803,
        "stage_b_seeds": [20260803, 20260804, 20260805],
        "sources": sorted(SOURCES),
        "forbidden_sources": sorted(FORBIDDEN_SOURCES),
        "split_sha256": SPLIT_SHA256,
        "parameter_cap": 100000,
        "external_data_accessed": False,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--manifest", type=Path, default=Path("results/experiment0/record_manifest.csv"))
    parser.add_argument("--splits", type=Path, default=Path("results/experiment1/splits.json"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiment2_6"))
    parser.add_argument("--window-cache", type=Path, default=Path("results/experiment2_6/window_cache"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 20 <= args.epochs <= 30:
        parser.error("Experiment 2.6 requires a preregistered maximum in the 20-30 epoch range")
    payload = json.loads(args.splits.read_text(encoding="utf-8"))
    if payload.get("split_sha256") != SPLIT_SHA256:
        raise ValueError("Experiment 1 split hash is not the reviewed value")
    condition = payload["conditions"].get("M1")
    if condition is None or set(condition["sources"]) != SOURCES:
        raise ValueError("Experiment 2.6 is restricted to MIT-BIH plus INCART")
    if not 1 <= args.outer_fold <= len(condition["outer_folds"]):
        parser.error("--outer-fold is outside the frozen M1 split manifest")
    records = [record for record in core.prep.load_training_records(args.manifest) if record.source in SOURCES]
    if {record.source for record in records} != SOURCES:
        raise ValueError("M1 training manifest is incomplete")
    test_keys = set(condition["outer_folds"][args.outer_fold - 1]["records"])
    development = [record for record in records if record.key not in test_keys]
    test = [record for record in records if record.key in test_keys]
    if {record.key for record in development} & {record.key for record in test}:
        raise AssertionError("Outer-fold record leakage")
    write_protocol(args.output_dir)
    if args.dry_run:
        print(json.dumps({"status": "passed", "candidate": args.candidate, "outer_fold": args.outer_fold, "sources": sorted(SOURCES), "external_data_accessed": False}, indent=2))
        return
    store_class = RobustRRStore if args.candidate == "P2" else core.RecordStore
    store = store_class(records, args.data_root, args.window_cache)
    oof, curves, chosen_epochs = {}, {}, []
    inner_folds = [fold for index, fold in enumerate(condition["outer_folds"]) if index != args.outer_fold - 1]
    for inner_index, fold in enumerate(inner_folds):
        validation_keys = set(fold["records"])
        inner_train = [record for record in development if record.key not in validation_keys]
        inner_validation = [record for record in development if record.key in validation_keys]
        model, selected_epoch, curve = train_select_epoch(args.candidate, inner_train, inner_validation, store, args, args.seed + inner_index)
        oof.update(core.predict_records(model, "O1", inner_validation, store))
        curves[f"inner_{inner_index + 1}"] = curve
        chosen_epochs.append(selected_epoch)
    scaler, threshold, selection = calibrate_and_select(oof, args.candidate)
    final_epochs = int(np.median(chosen_epochs))
    final_model = train_epochs(args.candidate, development, store, args, args.seed + 100, final_epochs)
    test_raw = core.predict_records(final_model, "O1", test, store)
    test_calibrated = {key: (labels, apply_scaler(scaler, probabilities, args.candidate)) for key, (labels, probabilities) in test_raw.items()}
    per_record, test_summary = core.aggregate_records(test_calibrated, threshold)
    prediction_dir = args.output_dir / "per_record_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for key, (labels, probabilities) in test_calibrated.items():
        np.savez_compressed(prediction_dir / f"{args.candidate}_fold{args.outer_fold}_seed{args.seed}_{key.replace(':', '_')}.npz", labels=labels, calibrated_probability=probabilities)
    curve_dir = args.output_dir / "learning_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    (curve_dir / f"{args.candidate}_fold{args.outer_fold}_seed{args.seed}.json").write_text(json.dumps(curves, indent=2) + "\n", encoding="utf-8")
    architecture = final_model.to_json()
    result = {
        "experiment": "experiment_2_6_controlled_refinement",
        "stage": "A" if args.seed == 20260803 else "B",
        "candidate": args.candidate,
        "outer_fold": args.outer_fold,
        "seed": args.seed,
        "split_sha256": SPLIT_SHA256,
        "sources": sorted(SOURCES),
        "lead_profile": "L2",
        "input_shape": [300, 2],
        "external_data_accessed": False,
        "training": {"max_epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size, "final_epochs": final_epochs, "inner_selected_epochs": chosen_epochs, "sampler": "hierarchical source -> record -> class"},
        "selection": selection | {"threshold": threshold, "epoch_selection_metric": "inner record-macro AUPRC"},
        "architecture": {"parameter_count": int(final_model.count_params()), "json_sha256": hashlib.sha256(architecture.encode("utf-8")).hexdigest()},
        "outer_test": test_summary,
        "per_record": per_record,
        "source_code_sha256": sha256_file(Path(__file__)),
    }
    path = args.output_dir / f"{args.candidate}_fold{args.outer_fold}_seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
