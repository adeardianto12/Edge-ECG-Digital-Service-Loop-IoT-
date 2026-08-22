"""Run one M1 lead-profile ablation fold for Experiment 2.

The program uses only the frozen Experiment 1 M1 record groups.  It neither
enumerates nor accepts SVDB/NSRDB, and it does not alter Experiment 1 outputs.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from multisource_ecg import SOURCE_BY_KEY, load_pvc_windows


EXPECTED_SPLIT_SHA256 = "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787"
SELECTED_SOURCES = {"mitdb", "incartdb"}
FORBIDDEN_SOURCES = {"svdb", "nsrdb"}
LEAD_PROFILES = {"L1": 1, "L2": 2}


def load_experiment1_prepare():
    path = Path(__file__).with_name("08_experiment1_prepare.py")
    spec = importlib.util.spec_from_file_location("experiment1_prepare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Experiment 1 preparation module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prep = load_experiment1_prepare()


class LeadWindowSampler:
    """Source -> record -> class sampler with a lead-profile-specific cache."""

    def __init__(
        self, records: list, data_root: Path, seed: int, lead_count: int, cache_dir: Path
    ) -> None:
        self.records = records
        self.data_root = data_root
        self.rng = np.random.default_rng(seed)
        self.lead_count = lead_count
        self.cache_dir = cache_dir
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.cache_size = len(records) if {item.source for item in records} == {"mitdb"} else 8
        self.by_source_class: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        for record in records:
            if record.source not in SELECTED_SOURCES or record.source in FORBIDDEN_SOURCES:
                raise ValueError(f"Experiment 2 source violation: {record.source}")
            if record.normal_count:
                self.by_source_class[record.source][0].append(record)
            if record.pvc_count:
                self.by_source_class[record.source][1].append(record)

    def _load(self, record) -> dict:
        if record.key in self.cache:
            self.cache.move_to_end(record.key)
            return self.cache[record.key]
        record_dir = self.cache_dir / record.source
        waveforms_path = record_dir / f"{record.record}_waveforms.npy"
        labels_path = record_dir / f"{record.record}_labels.npy"
        if waveforms_path.exists() and labels_path.exists():
            loaded = {
                "waveforms": np.load(waveforms_path, mmap_mode="r"),
                "labels": np.load(labels_path, mmap_mode="r"),
            }
            if loaded["waveforms"].shape[2] != self.lead_count:
                raise ValueError(f"Cached lead count differs for {record.key}")
        else:
            loaded = load_pvc_windows(
                self.data_root / SOURCE_BY_KEY[record.source].directory / record.record,
                SOURCE_BY_KEY[record.source],
                lead_count=self.lead_count,
            )
            record_dir.mkdir(parents=True, exist_ok=True)
            np.save(waveforms_path, loaded["waveforms"])
            np.save(labels_path, loaded["labels"])
        self.cache[record.key] = loaded
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return loaded

    def sample_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        eligible_sources = [
            source
            for source, by_class in self.by_source_class.items()
            if by_class[0] and by_class[1]
        ]
        source = eligible_sources[int(self.rng.integers(len(eligible_sources)))]
        windows, labels = [], []
        for label, count in ((0, batch_size // 2), (1, batch_size - batch_size // 2)):
            candidates = self.by_source_class[source][label]
            record = candidates[int(self.rng.integers(len(candidates)))]
            loaded = self._load(record)
            positions = np.flatnonzero(loaded["labels"] == label)
            if not len(positions):
                raise RuntimeError(f"Manifest and cache disagree for {record.key}, class {label}")
            selected = positions[self.rng.integers(len(positions), size=count)]
            windows.extend(loaded["waveforms"][int(index)] for index in selected)
            labels.extend([label] * count)
        order = self.rng.permutation(batch_size)
        return np.stack([windows[index] for index in order]), np.asarray([labels[index] for index in order], dtype=np.int32)


def build_model(lead_count: int):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(300, lead_count), name="ecg_beat")
    x = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="class")(x)
    model = tf.keras.Model(inputs, outputs, name=f"experiment2_L{lead_count}_waveform")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.Recall(class_id=1, name="pvc_recall")],
    )
    return model


def train(records: list, args, seed_offset: int):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed + seed_offset)
    sampler = LeadWindowSampler(records, args.data_root, args.seed + seed_offset, args.lead_count, args.window_cache)
    model = build_model(args.lead_count)
    for _ in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            windows, labels = sampler.sample_batch(args.batch_size)
            model.train_on_batch(windows, labels)
    return model


def record_windows(record, args) -> dict:
    sampler = LeadWindowSampler([record], args.data_root, 0, args.lead_count, args.window_cache)
    return sampler._load(record)


def predict_records(model, records: list, args) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for record in records:
        loaded = record_windows(record, args)
        if len(loaded["labels"]):
            result[record.key] = (
                loaded["labels"],
                model.predict(loaded["waveforms"], batch_size=1024, verbose=0)[:, 1],
            )
    return result


def apply_calibration(scaler: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(probabilities / (1 - probabilities))
    return scaler.predict_proba(logits.reshape(-1, 1))[:, 1]


def calibrate(predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[LogisticRegression, float, dict]:
    labels = np.concatenate([item[0] for item in predictions.values()])
    probabilities = np.concatenate([item[1] for item in predictions.values()])
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    scaler = LogisticRegression(random_state=0).fit(np.log(clipped / (1 - clipped)).reshape(-1, 1), labels)
    calibrated_by_record = {
        key: apply_calibration(scaler, probabilities) for key, (_, probabilities) in predictions.items()
    }
    candidates = []
    for threshold in np.arange(0.01, 1.0, 0.01):
        decisions = np.concatenate([item >= threshold for item in calibrated_by_record.values()])
        recall = float(((decisions == 1) & (labels == 1)).sum() / max((labels == 1).sum(), 1))
        record_f1 = []
        for key, (record_labels, _) in predictions.items():
            record_decisions = calibrated_by_record[key] >= threshold
            tp = ((record_decisions == 1) & (record_labels == 1)).sum()
            fp = ((record_decisions == 1) & (record_labels == 0)).sum()
            fn = ((record_decisions == 0) & (record_labels == 1)).sum()
            record_f1.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
        candidates.append((recall >= 0.90, float(np.mean(record_f1)), recall, float(threshold)))
    selected = max([item for item in candidates if item[0]] or candidates, key=lambda item: (item[1], item[2], item[3]))
    return scaler, selected[3], {"recall_target_met": selected[0], "record_macro_pvc_f1": selected[1], "pvc_recall": selected[2]}


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    decision = probabilities >= threshold
    tp = int(((decision == 1) & (labels == 1)).sum()); fp = int(((decision == 1) & (labels == 0)).sum())
    tn = int(((decision == 0) & (labels == 0)).sum()); fn = int(((decision == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    both_classes = len(np.unique(labels)) == 2
    return {
        "pvc_precision": precision,
        "pvc_recall": recall,
        "pvc_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "auroc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "auprc": float(average_precision_score(labels, probabilities)) if both_classes else None,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-profile", choices=tuple(LEAD_PROFILES), required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pilot", action="store_true", help="Mark a reduced protocol check as non-official.")
    parser.add_argument("--manifest", type=Path, default=Path("results/experiment0/record_manifest.csv"))
    parser.add_argument("--splits", type=Path, default=Path("results/experiment1/splits.json"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiment2"))
    parser.add_argument("--window-cache", type=Path, default=Path("results/experiment2/window_cache"))
    args = parser.parse_args()
    args.lead_count = LEAD_PROFILES[args.lead_profile]
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    if split_payload.get("split_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError("Experiment 1 split hash is not the reviewed frozen value")
    condition = split_payload["conditions"].get("M1")
    if condition is None or set(condition["sources"]) != SELECTED_SOURCES:
        raise ValueError("Experiment 2 must use the reviewed M1 sources only")
    if not 1 <= args.outer_fold <= len(condition["outer_folds"]):
        parser.error("--outer-fold is outside the frozen M1 split manifest")

    records = [item for item in prep.load_training_records(args.manifest) if item.source in SELECTED_SOURCES]
    if {item.source for item in records} != SELECTED_SOURCES:
        raise ValueError("M1 record manifest is incomplete")
    test_keys = set(condition["outer_folds"][args.outer_fold - 1]["records"])
    development = [item for item in records if item.key not in test_keys]
    test = [item for item in records if item.key in test_keys]
    if {item.key for item in development} & {item.key for item in test}:
        raise AssertionError("Outer-fold record leakage")

    # Four development folds produce calibration and threshold evidence without outer-test access.
    oof = {}
    for inner_index, fold in enumerate(
        item for index, item in enumerate(condition["outer_folds"]) if index != args.outer_fold - 1
    ):
        calibration_keys = set(fold["records"])
        model = train([item for item in development if item.key not in calibration_keys], args, inner_index)
        oof.update(predict_records(model, [item for item in development if item.key in calibration_keys], args))
    scaler, threshold, selection = calibrate(oof)
    final_model = train(development, args, 100)
    test_predictions = predict_records(final_model, test, args)

    per_record = {}
    for key, (labels, probabilities) in test_predictions.items():
        calibrated = apply_calibration(scaler, probabilities)
        per_record[key] = metrics(labels, calibrated, threshold)
        prediction_path = args.output_dir / "per_record_predictions" / f"{args.lead_profile}_fold{args.outer_fold}_seed{args.seed}_{key.replace(':', '_')}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, labels=labels, calibrated_probability=calibrated)
    labels = np.concatenate([item[0] for item in test_predictions.values()])
    probabilities = np.concatenate([apply_calibration(scaler, item[1]) for item in test_predictions.values()])
    architecture = final_model.to_json()
    result = {
        "experiment": "experiment_2_one_vs_two_channel_ablation",
        "status": "pilot" if args.pilot else "complete",
        "official_result": not args.pilot,
        "lead_profile": args.lead_profile,
        "lead_count": args.lead_count,
        "input_shape": [300, args.lead_count],
        "outer_fold": args.outer_fold,
        "seed": args.seed,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "training": {"epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size, "sources": sorted(SELECTED_SOURCES)},
        "selection": selection | {"threshold": threshold, "calibration": "Platt scaling fitted on inner out-of-fold record predictions"},
        "architecture": {"parameter_count": int(final_model.count_params()), "sha256": hashlib.sha256(architecture.encode("utf-8")).hexdigest()},
        "resource_measurement": {"float_model_size_bytes": int(final_model.count_params() * 4), "int8_model_size_bytes": None, "peak_ram_bytes": None, "target_device_latency_ms": None, "status": "deferred_to_experiment_7", "reason": "No declared target device or integer export exists; desktop timing is not reported as edge latency."},
        "outer_test": metrics(labels, probabilities, threshold),
        "per_record": per_record,
        "source_code_sha256": file_sha256(Path(__file__)),
        "external_data_accessed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.lead_profile}_fold{args.outer_fold}_seed{args.seed}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
