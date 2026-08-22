"""Run one preregistered Experiment 2.5 M1/L2 outer-fold candidate.

Only MIT-BIH and INCART development records are accepted.  The program has no
SVDB/NSRDB code path and writes separate Experiment 2.5 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import wfdb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from multisource_ecg import SOURCE_BY_KEY, TARGET_SAMPLE_RATE_HZ, load_pvc_windows


SPLIT_SHA256 = "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787"
SOURCES = {"mitdb", "incartdb"}
FORBIDDEN_SOURCES = {"svdb", "nsrdb"}
CANDIDATES = ("O0", "O1", "O2", "O3")


def load_prepare_module():
    path = Path(__file__).with_name("08_experiment1_prepare.py")
    spec = importlib.util.spec_from_file_location("experiment1_prepare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Experiment 1 preparation module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prep = load_prepare_module()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_from_key(key: str) -> str:
    return key.split(":", 1)[0]


class RecordStore:
    """M1/L2 cache with causal RR features derived from past annotation peaks."""

    def __init__(self, records: list, data_root: Path, cache_dir: Path, lead_count: int = 2) -> None:
        self.records = records
        self.data_root = data_root
        self.cache_dir = cache_dir
        self.lead_count = lead_count
        self.memory: OrderedDict[str, dict] = OrderedDict()
        self.cache_size = 8
        self.by_source_class: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        for record in records:
            if record.source not in SOURCES or record.source in FORBIDDEN_SOURCES:
                raise ValueError(f"Experiment 2.5 source violation: {record.source}")
            if record.normal_count:
                self.by_source_class[record.source][0].append(record)
            if record.pvc_count:
                self.by_source_class[record.source][1].append(record)

    def _rr_features(self, record) -> np.ndarray:
        """Use only preceding annotated peaks; labels are not used for RR history."""
        record_base = self.data_root / SOURCE_BY_KEY[record.source].directory / record.record
        header = wfdb.rdheader(str(record_base))
        annotation = wfdb.rdann(str(record_base), "atr")
        all_peaks = np.unique(np.rint(annotation.sample * TARGET_SAMPLE_RATE_HZ / header.fs).astype(np.int64))
        accepted = load_pvc_windows(record_base, SOURCE_BY_KEY[record.source], lead_count=self.lead_count)
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
        return np.asarray(output, dtype=np.float32)

    def load(self, record) -> dict:
        if record.key in self.memory:
            self.memory.move_to_end(record.key)
            return self.memory[record.key]
        directory = self.cache_dir / record.source
        waveform_path = directory / f"{record.record}_waveforms.npy"
        label_path = directory / f"{record.record}_labels.npy"
        rr_path = directory / f"{record.record}_causal_rr.npy"
        if waveform_path.exists() and label_path.exists() and rr_path.exists():
            loaded = {
                "waveforms": np.load(waveform_path, mmap_mode="r"),
                "labels": np.load(label_path, mmap_mode="r"),
                "rr": np.load(rr_path, mmap_mode="r"),
            }
        else:
            directory.mkdir(parents=True, exist_ok=True)
            raw = load_pvc_windows(
                self.data_root / SOURCE_BY_KEY[record.source].directory / record.record,
                SOURCE_BY_KEY[record.source],
                lead_count=self.lead_count,
            )
            loaded = {"waveforms": raw["waveforms"], "labels": raw["labels"], "rr": self._rr_features(record)}
            np.save(waveform_path, loaded["waveforms"])
            np.save(label_path, loaded["labels"])
            np.save(rr_path, loaded["rr"])
        if loaded["waveforms"].shape[2] != self.lead_count or len(loaded["labels"]) != len(loaded["rr"]):
            raise ValueError(f"Corrupt Experiment 2.5 cache for {record.key}")
        self.memory[record.key] = loaded
        while len(self.memory) > self.cache_size:
            self.memory.popitem(last=False)
        return loaded


class HierarchicalSampler:
    def __init__(self, records: list, store: RecordStore, seed: int, augment: bool) -> None:
        self.store = store
        self.rng = np.random.default_rng(seed)
        self.augment = augment
        self.by_source_class: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        for record in records:
            if record.normal_count:
                self.by_source_class[record.source][0].append(record)
            if record.pvc_count:
                self.by_source_class[record.source][1].append(record)

    def _augment(self, windows: np.ndarray) -> np.ndarray:
        result = windows.copy()
        for index, window in enumerate(result):
            if self.rng.random() < 0.35:
                shift = int(self.rng.integers(-18, 19))
                window[:] = np.roll(window, shift, axis=0)
            time = np.arange(window.shape[0], dtype=np.float32) / TARGET_SAMPLE_RATE_HZ
            if self.rng.random() < 0.30:
                window += self.rng.uniform(0.01, 0.05) * np.sin(2 * np.pi * self.rng.uniform(0.15, 0.5) * time)[:, None]
            if self.rng.random() < 0.25:
                window += self.rng.uniform(0.005, 0.02) * np.sin(2 * np.pi * self.rng.choice((50.0, 60.0)) * time)[:, None]
            if self.rng.random() < 0.25:
                rms = np.sqrt(np.mean(np.square(window), axis=0, keepdims=True))
                snr = self.rng.uniform(25.0, 35.0)
                window += self.rng.normal(0.0, rms / (10 ** (snr / 20)), size=window.shape)
            if self.rng.random() < 0.05:
                window[:, int(self.rng.integers(window.shape[1]))] = 0.0
        return result

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eligible = [source for source, values in self.by_source_class.items() if values[0] and values[1]]
        source = eligible[int(self.rng.integers(len(eligible)))]
        windows, rr, labels = [], [], []
        for label, count in ((0, batch_size // 2), (1, batch_size - batch_size // 2)):
            record = self.by_source_class[source][label][int(self.rng.integers(len(self.by_source_class[source][label])))]
            loaded = self.store.load(record)
            positions = np.flatnonzero(loaded["labels"] == label)
            choice = positions[self.rng.integers(len(positions), size=count)]
            windows.extend(loaded["waveforms"][int(item)] for item in choice)
            rr.extend(loaded["rr"][int(item)] for item in choice)
            labels.extend([label] * count)
        order = self.rng.permutation(batch_size)
        waveforms = np.stack([windows[item] for item in order]).astype(np.float32)
        if self.augment:
            waveforms = self._augment(waveforms)
        return waveforms, np.stack([rr[item] for item in order]).astype(np.float32), np.asarray([labels[item] for item in order], dtype=np.int32)


def rr_branch(tf):
    inputs = tf.keras.Input(shape=(5,), name="causal_rr")
    x = tf.keras.layers.Dense(8, activation="relu", name="rr_mlp")(inputs)
    return inputs, x


def residual_block(tf, x, filters: int, dilation: int, name: str):
    shortcut = x
    x = tf.keras.layers.SeparableConv1D(filters, 9, padding="same", dilation_rate=dilation, use_bias=False, name=f"{name}_sep1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu1")(x)
    x = tf.keras.layers.SeparableConv1D(filters, 9, padding="same", dilation_rate=dilation, use_bias=False, name=f"{name}_sep2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding="same", use_bias=False, name=f"{name}_skip")(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)
    x = tf.keras.layers.Add(name=f"{name}_add")([x, shortcut])
    return tf.keras.layers.ReLU(name=f"{name}_out")(x)


def build_model(candidate: str):
    import tensorflow as tf

    waveform = tf.keras.Input(shape=(300, 2), name="ecg_beat")
    if candidate in {"O0", "O1"}:
        x = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu")(waveform)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(x)
        morphology = tf.keras.layers.GlobalAveragePooling1D()(x)
    else:
        x = tf.keras.layers.Conv1D(24, 7, padding="same", use_bias=False, name="stem")(waveform)
        x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
        x = tf.keras.layers.ReLU(name="stem_relu")(x)
        for index, dilation in enumerate((1, 2, 4, 8), start=1):
            x = residual_block(tf, x, 24, dilation, f"block{index}")
        morphology = tf.keras.layers.Concatenate(name="morphology_pool")([
            tf.keras.layers.GlobalAveragePooling1D()(x), tf.keras.layers.GlobalMaxPooling1D()(x)
        ])
    inputs = [waveform]
    if candidate != "O0":
        rr_inputs, rr_features = rr_branch(tf)
        morphology = tf.keras.layers.Concatenate(name="fusion")([morphology, rr_features])
        inputs.append(rr_inputs)
    x = tf.keras.layers.Dense(32 if candidate in {"O2", "O3"} else 16, activation="relu", name="classifier_hidden")(morphology)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="class")(x)
    model = tf.keras.Model(inputs, outputs, name=f"experiment2_5_{candidate}")
    if model.count_params() >= 100_000:
        raise AssertionError("Experiment 2.5 architecture cap exceeded")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def model_inputs(candidate: str, waveforms: np.ndarray, rr: np.ndarray):
    return waveforms if candidate == "O0" else [waveforms, rr]


def predict_records(model, candidate: str, records: list, store: RecordStore) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for record in records:
        loaded = store.load(record)
        if len(loaded["labels"]):
            probabilities = model.predict(model_inputs(candidate, loaded["waveforms"], loaded["rr"]), batch_size=1024, verbose=0)[:, 1]
            output[record.key] = (np.asarray(loaded["labels"]), probabilities)
    return output


def per_record_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    decision = probabilities >= threshold
    tp = int(np.sum(decision & (labels == 1))); fp = int(np.sum(decision & (labels == 0)))
    tn = int(np.sum(~decision & (labels == 0))); fn = int(np.sum(~decision & (labels == 1)))
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    has_pvc = bool(np.any(labels == 1)); both = len(np.unique(labels)) == 2
    return {
        "eligible_N": int(np.sum(labels == 0)), "eligible_V": int(np.sum(labels == 1)), "has_pvc_reference": has_pvc,
        "pvc_precision": precision if has_pvc else None, "pvc_recall": recall if has_pvc else None,
        "pvc_f1": (2 * precision * recall / max(precision + recall, 1e-12)) if has_pvc else 0.0,
        "specificity": tn / max(tn + fp, 1), "auroc": float(roc_auc_score(labels, probabilities)) if both else None,
        "auprc": float(average_precision_score(labels, probabilities)) if both else None,
        "brier_score": float(brier_score_loss(labels, probabilities)), "false_pvc_decisions": fp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def aggregate_records(predictions: dict[str, tuple[np.ndarray, np.ndarray]], threshold: float) -> tuple[dict, dict]:
    per_record = {key: per_record_metrics(labels, probabilities, threshold) for key, (labels, probabilities) in predictions.items()}
    labels = np.concatenate([value[0] for value in predictions.values()])
    probabilities = np.concatenate([value[1] for value in predictions.values()])
    pooled = per_record_metrics(labels, probabilities, threshold)
    support = [item for item in per_record.values() if item["has_pvc_reference"]]
    pvc_free = [item for item in per_record.values() if not item["has_pvc_reference"]]
    def mean_defined(items, field):
        values = [item[field] for item in items if item[field] is not None]
        return float(np.mean(values)) if values else None
    record_summary = {
        "support_aware": {"record_count": len(support), "pvc_precision": mean_defined(support, "pvc_precision"), "pvc_recall": mean_defined(support, "pvc_recall"), "pvc_f1": mean_defined(support, "pvc_f1")},
        "legacy_zero_filled_macro_pvc_f1": float(np.mean([item["pvc_f1"] for item in per_record.values()])),
        "record_macro_auprc": mean_defined([item for item in per_record.values() if item["auprc"] is not None], "auprc"),
        "record_macro_brier_score": mean_defined(list(per_record.values()), "brier_score"),
        "pvc_free_records": {"record_count": len(pvc_free), "false_pvc_decisions": int(sum(item["false_pvc_decisions"] for item in pvc_free)), "mean_specificity": mean_defined(pvc_free, "specificity")},
    }
    return per_record, {"pooled": pooled, "record_macro": record_summary}


def balanced_sample_weights(predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    source_counts = defaultdict(int)
    for key in predictions:
        source_counts[source_from_key(key)] += 1
    weights = []
    for key, (labels, _) in predictions.items():
        weights.extend([1.0 / (source_counts[source_from_key(key)] * max(len(labels), 1))] * len(labels))
    return np.asarray(weights, dtype=np.float64)


def calibrate_and_select(predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[LogisticRegression, float, dict]:
    labels = np.concatenate([value[0] for value in predictions.values()])
    raw = np.concatenate([value[1] for value in predictions.values()])
    logits = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
    scaler = LogisticRegression(random_state=0, max_iter=1000).fit(logits.reshape(-1, 1), labels, sample_weight=balanced_sample_weights(predictions))
    calibrated = {}
    cursor = 0
    transformed = scaler.predict_proba(logits.reshape(-1, 1))[:, 1]
    for key, (record_labels, _) in predictions.items():
        calibrated[key] = (record_labels, transformed[cursor:cursor + len(record_labels)])
        cursor += len(record_labels)
    options = []
    for threshold in np.arange(0.01, 1.00, 0.01):
        _, summary = aggregate_records(calibrated, float(threshold))
        pooled_recall = summary["pooled"]["pvc_recall"]
        support = summary["record_macro"]["support_aware"]
        options.append({"threshold": float(threshold), "pooled_recall": pooled_recall, "support_aware_recall": support["pvc_recall"], "support_aware_f1": support["pvc_f1"], "pvc_free_false_decisions": summary["record_macro"]["pvc_free_records"]["false_pvc_decisions"]})
    feasible = [item for item in options if item["pooled_recall"] >= 0.90 and item["support_aware_recall"] >= 0.90]
    selected = dict(max(feasible or options, key=lambda item: (item["support_aware_f1"], item["pooled_recall"], -item["pvc_free_false_decisions"], -item["threshold"])))
    selected["recall_target_met"] = bool(feasible)
    selected["calibration"] = "Platt scaling with source- and record-balanced fitting weights"
    selected["threshold_trace"] = options
    return scaler, float(selected["threshold"]), selected


def apply_scaler(scaler: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))
    return scaler.predict_proba(logits.reshape(-1, 1))[:, 1]


def train_select_epoch(candidate: str, train_records: list, validation_records: list, store: RecordStore, args, seed: int):
    import tensorflow as tf
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    sampler = HierarchicalSampler(train_records, store, seed, augment=candidate == "O3")
    model = build_model(candidate)
    curve, best_epoch, best_score, best_weights = [], 1, -np.inf, None
    for epoch in range(1, args.epochs + 1):
        for _ in range(args.steps_per_epoch):
            waveforms, rr, labels = sampler.sample(args.batch_size)
            model.train_on_batch(model_inputs(candidate, waveforms, rr), labels)
        predictions = predict_records(model, candidate, validation_records, store)
        _, metrics = aggregate_records(predictions, 0.5)
        score = metrics["record_macro"]["record_macro_auprc"]
        curve.append({"epoch": epoch, "selection_record_macro_auprc": score})
        if score is not None and score > best_score:
            best_epoch, best_score, best_weights = epoch, score, model.get_weights()
    if best_weights is None:
        raise RuntimeError("Inner epoch selection did not produce a usable score")
    model.set_weights(best_weights)
    return model, best_epoch, curve


def train_fixed_epochs(candidate: str, records: list, store: RecordStore, args, seed: int, epochs: int):
    import tensorflow as tf
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    sampler = HierarchicalSampler(records, store, seed, augment=candidate == "O3")
    model = build_model(candidate)
    for _ in range(epochs):
        for _ in range(args.steps_per_epoch):
            waveforms, rr, labels = sampler.sample(args.batch_size)
            model.train_on_batch(model_inputs(candidate, waveforms, rr), labels)
    return model


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
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiment2_5"))
    parser.add_argument("--window-cache", type=Path, default=Path("results/experiment2_5/window_cache"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 20 <= args.epochs <= 30:
        parser.error("Experiment 2.5 requires a preregistered maximum in the 20-30 epoch range")
    if args.batch_size < 2 or args.steps_per_epoch < 1:
        parser.error("Batch size and steps per epoch must be positive")
    payload = json.loads(args.splits.read_text(encoding="utf-8"))
    if payload.get("split_sha256") != SPLIT_SHA256:
        raise ValueError("Experiment 1 split hash is not the reviewed value")
    condition = payload["conditions"].get("M1")
    if condition is None or set(condition["sources"]) != SOURCES:
        raise ValueError("Experiment 2.5 is restricted to M1 MIT-BIH plus INCART")
    if not 1 <= args.outer_fold <= len(condition["outer_folds"]):
        parser.error("--outer-fold is outside the frozen M1 split manifest")
    records = [record for record in prep.load_training_records(args.manifest) if record.source in SOURCES]
    if {record.source for record in records} != SOURCES:
        raise ValueError("M1 training manifest is incomplete")
    test_keys = set(condition["outer_folds"][args.outer_fold - 1]["records"])
    development = [record for record in records if record.key not in test_keys]
    test = [record for record in records if record.key in test_keys]
    if {record.key for record in development} & {record.key for record in test}:
        raise AssertionError("Outer-fold record leakage")
    if args.dry_run:
        print(json.dumps({"status": "passed", "candidate": args.candidate, "outer_fold": args.outer_fold, "sources": sorted(SOURCES), "development_records": len(development), "test_records": len(test), "external_data_accessed": False}, indent=2))
        return
    store = RecordStore(records, args.data_root, args.window_cache)
    oof, curves, chosen_epochs = {}, {}, []
    inner_folds = [fold for index, fold in enumerate(condition["outer_folds"]) if index != args.outer_fold - 1]
    for inner_index, fold in enumerate(inner_folds):
        validation_keys = set(fold["records"])
        inner_train = [record for record in development if record.key not in validation_keys]
        inner_validation = [record for record in development if record.key in validation_keys]
        model, selected_epoch, curve = train_select_epoch(args.candidate, inner_train, inner_validation, store, args, args.seed + inner_index)
        oof.update(predict_records(model, args.candidate, inner_validation, store))
        curves[f"inner_{inner_index + 1}"] = curve
        chosen_epochs.append(selected_epoch)
    scaler, threshold, selection = calibrate_and_select(oof)
    final_epochs = int(np.median(chosen_epochs))
    final_model = train_fixed_epochs(args.candidate, development, store, args, args.seed + 100, final_epochs)
    test_raw = predict_records(final_model, args.candidate, test, store)
    test_calibrated = {key: (labels, apply_scaler(scaler, probabilities)) for key, (labels, probabilities) in test_raw.items()}
    per_record, test_summary = aggregate_records(test_calibrated, threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "per_record_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for key, (labels, probabilities) in test_calibrated.items():
        np.savez_compressed(prediction_dir / f"{args.candidate}_fold{args.outer_fold}_seed{args.seed}_{key.replace(':', '_')}.npz", labels=labels, calibrated_probability=probabilities)
    curve_path = args.output_dir / "learning_curves"
    curve_path.mkdir(parents=True, exist_ok=True)
    (curve_path / f"{args.candidate}_fold{args.outer_fold}_seed{args.seed}.json").write_text(json.dumps(curves, indent=2) + "\n", encoding="utf-8")
    architecture = final_model.to_json()
    result = {
        "experiment": "experiment_2_5_pre_freeze_model_optimization", "status": "stage_a" if args.seed == 20260803 else "stage_b", "official_result": True,
        "candidate": args.candidate, "outer_fold": args.outer_fold, "seed": args.seed, "split_sha256": SPLIT_SHA256,
        "sources": sorted(SOURCES), "lead_profile": "L2", "input_shape": [300, 2], "external_data_accessed": False,
        "training": {"max_epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size, "final_epochs": final_epochs, "inner_selected_epochs": chosen_epochs, "sampler": "hierarchical source -> record -> class", "augmentation": args.candidate == "O3"},
        "selection": selection | {"threshold": threshold, "epoch_selection_metric": "inner record-macro AUPRC"},
        "architecture": {"parameter_count": int(final_model.count_params()), "json_sha256": hashlib.sha256(architecture.encode("utf-8")).hexdigest()},
        "outer_test": test_summary, "per_record": per_record,
        "source_code_sha256": sha256_file(Path(__file__)),
    }
    path = args.output_dir / f"{args.candidate}_fold{args.outer_fold}_seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
