"""Prepare and validate Experiment 1 multi-source training inputs.

This script creates the record-level split manifest and exercises the
source -> record -> class sampler before any model fitting.  It deliberately
does not open SVDB or NSRDB files: those sources are forbidden from every
Experiment 1 code path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from multisource_ecg import SOURCE_BY_KEY, load_pvc_windows


CONDITIONS = {
    "M0": ("mitdb",),
    "M1": ("mitdb", "incartdb"),
    "M2": ("mitdb", "incartdb", "ltdb"),
}
FORBIDDEN_SOURCES = {"svdb", "nsrdb"}
FOLD_COUNT = 5
SPLIT_SEED = 20260803


@dataclass(frozen=True)
class Record:
    source: str
    record: str
    split_group: str
    normal_count: int
    pvc_count: int

    @property
    def key(self) -> str:
        return f"{self.source}:{self.record}"


def stable_hash(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def load_training_records(manifest_path: Path) -> list[Record]:
    """Read the Experiment 0 manifest while enforcing its source-role contract."""
    records: list[Record] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = row["source"]
            role = row["role"]
            if source in FORBIDDEN_SOURCES:
                if role == "training_development":
                    raise ValueError(f"Forbidden source has training role: {source}")
                continue
            if role != "training_development":
                continue
            records.append(
                Record(
                    source=source,
                    record=row["record"],
                    split_group=row["split_group"],
                    normal_count=int(row["eligible_N"]),
                    pvc_count=int(row["eligible_V"]),
                )
            )
    if not records:
        raise ValueError("No training-development records found in the Experiment 0 manifest")
    return records


def group_records(records: list[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.split_group].append(record)
    for group, members in grouped.items():
        sources = {member.source for member in members}
        if len(sources) != 1:
            raise ValueError(f"Split group mixes sources: {group}")
    return dict(grouped)


def group_summary(members: list[Record]) -> dict:
    return {
        "split_group": members[0].split_group,
        "source": members[0].source,
        "records": [member.record for member in sorted(members, key=lambda item: item.record)],
        "eligible_N": sum(member.normal_count for member in members),
        "eligible_V": sum(member.pvc_count for member in members),
    }


def allocate_source_groups(groups: list[dict], fold_count: int) -> list[list[dict]]:
    """Greedily balance N/V load within one source without splitting records."""
    if len(groups) < fold_count:
        raise ValueError("Every selected source must provide at least one group per outer fold")
    target_n = sum(group["eligible_N"] for group in groups) / fold_count
    target_v = sum(group["eligible_V"] for group in groups) / fold_count
    folds: list[list[dict]] = [[] for _ in range(fold_count)]
    totals = np.zeros((fold_count, 3), dtype=np.float64)  # group count, N, V

    ordered_groups = sorted(
        groups, key=lambda item: (item["eligible_V"], item["eligible_N"]), reverse=True
    )
    # Seed every fold with one group.  This is required so that each outer
    # fold contains every selected source, even for LTDDB with only seven
    # record groups.
    for fold_index, group in enumerate(ordered_groups[:fold_count]):
        folds[fold_index].append(group)
        totals[fold_index] += (1, group["eligible_N"], group["eligible_V"])

    def global_score(candidate: np.ndarray) -> float:
        count_target = len(groups) / fold_count
        count_deviation = (candidate[:, 0] - count_target) / max(count_target, 1.0)
        normal_deviation = (candidate[:, 1] - target_n) / max(target_n, 1.0)
        pvc_deviation = (candidate[:, 2] - target_v) / max(target_v, 1.0)
        return float(
            0.25 * np.square(count_deviation).sum()
            + np.square(normal_deviation).sum()
            + 2.0 * np.square(pvc_deviation).sum()
        )

    for group in ordered_groups[fold_count:]:
        scores = []
        for fold_index in range(fold_count):
            candidate = totals.copy()
            candidate[fold_index] += (1, group["eligible_N"], group["eligible_V"])
            scores.append((global_score(candidate), totals[fold_index, 2], totals[fold_index, 1], fold_index))
        chosen = min(scores)[3]
        folds[chosen].append(group)
        totals[chosen] += (1, group["eligible_N"], group["eligible_V"])
    return folds


def build_condition_splits(grouped: dict[str, list[Record]], sources: tuple[str, ...]) -> dict:
    selected_groups = [group_summary(members) for members in grouped.values() if members[0].source in sources]
    unexpected = {group["source"] for group in selected_groups} - set(sources)
    if unexpected:
        raise ValueError(f"Unexpected source in condition: {sorted(unexpected)}")

    per_source: dict[str, list[list[dict]]] = {}
    for source in sources:
        source_groups = [group for group in selected_groups if group["source"] == source]
        per_source[source] = allocate_source_groups(source_groups, FOLD_COUNT)

    outer_folds = []
    for fold_index in range(FOLD_COUNT):
        groups = [group for source in sources for group in per_source[source][fold_index]]
        records = [f"{group['source']}:{record}" for group in groups for record in group["records"]]
        outer_folds.append(
            {
                "fold": fold_index + 1,
                "split_groups": [group["split_group"] for group in groups],
                "records": sorted(records),
                "source_counts": {
                    source: sum(1 for group in groups if group["source"] == source) for source in sources
                },
                "eligible_N": sum(group["eligible_N"] for group in groups),
                "eligible_V": sum(group["eligible_V"] for group in groups),
            }
        )

    return {
        "sources": list(sources),
        "outer_folds": outer_folds,
        "inner_protocol": {
            "method": "four-fold cross-fitting within each outer development set",
            "purpose": "epoch selection, Platt calibration, and threshold selection without using the outer test fold",
            "outer_fold_usage": "each outer fold is held out once; the remaining four outer folds define the inner record groups",
        },
    }


def verify_splits(payload: dict, grouped: dict[str, list[Record]]) -> dict:
    """Fail closed on source leakage, record leakage, or linked-record separation."""
    checks: list[dict] = []
    for condition, condition_payload in payload["conditions"].items():
        allowed_sources = set(condition_payload["sources"])
        seen_records: set[str] = set()
        for fold in condition_payload["outer_folds"]:
            fold_sources = {item.split(":", 1)[0] for item in fold["records"]}
            if fold_sources != allowed_sources:
                raise AssertionError(f"{condition} fold {fold['fold']} lacks a selected source")
            forbidden = fold_sources & FORBIDDEN_SOURCES
            if forbidden:
                raise AssertionError(f"{condition} includes forbidden sources: {sorted(forbidden)}")
            overlap = seen_records & set(fold["records"])
            if overlap:
                raise AssertionError(f"{condition} record appears in multiple outer folds: {sorted(overlap)}")
            seen_records.update(fold["records"])

        expected_records = {
            member.key
            for members in grouped.values()
            for member in members
            if member.source in allowed_sources
        }
        if seen_records != expected_records:
            raise AssertionError(f"{condition} records do not match its declared sources")

        linked = {member.key for member in grouped["mitdb:201_202"]}
        linked_fold_count = sum(linked.issubset(set(fold["records"])) for fold in condition_payload["outer_folds"])
        if linked_fold_count != 1:
            raise AssertionError(f"{condition} split MIT-BIH linked records 201/202")
        checks.append({"condition": condition, "record_count": len(seen_records), "status": "passed"})
    return {"status": "passed", "checks": checks}


def build_split_payload(manifest_path: Path) -> dict:
    records = load_training_records(manifest_path)
    grouped = group_records(records)
    payload = {
        "experiment": "experiment_1_multisource_training_ablation",
        "split_unit": "record-level split_group",
        "fold_count": FOLD_COUNT,
        "split_seed": SPLIT_SEED,
        "lead_profile": "one_channel",
        "prohibited_sources": sorted(FORBIDDEN_SOURCES),
        "conditions": {condition: build_condition_splits(grouped, sources) for condition, sources in CONDITIONS.items()},
    }
    payload["validation"] = verify_splits(payload, grouped)
    payload["split_sha256"] = stable_hash({key: value for key, value in payload.items() if key != "split_sha256"})
    return payload


class HierarchicalWindowSampler:
    """Uniformly samples source, record, class, then an ECG window.

    The small LRU cache bounds RAM while sampling the long LTDDB records.  It
    is intentionally not a whole-dataset concatenation, which would cause
    long records to dominate both memory and training updates.
    """

    def __init__(self, records: list[Record], data_root: Path, seed: int, cache_size: int = 2, window_cache: Path | None = None):
        self.rng = np.random.default_rng(seed)
        self.data_root = data_root
        self.cache_size = cache_size
        self.window_cache = window_cache
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.by_source_class: dict[str, dict[int, list[Record]]] = defaultdict(lambda: defaultdict(list))
        for record in records:
            if record.source in FORBIDDEN_SOURCES:
                raise ValueError(f"Forbidden source cannot be sampled: {record.source}")
            if record.normal_count:
                self.by_source_class[record.source][0].append(record)
            if record.pvc_count:
                self.by_source_class[record.source][1].append(record)
        self.sources = sorted(self.by_source_class)

    def _load(self, record: Record) -> dict:
        key = record.key
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        waveforms_path = labels_path = None
        if self.window_cache is not None:
            record_dir = self.window_cache / record.source
            waveforms_path = record_dir / f"{record.record}_waveforms.npy"
            labels_path = record_dir / f"{record.record}_labels.npy"
        if waveforms_path is not None and waveforms_path.exists() and labels_path.exists():
            loaded = {"waveforms": np.load(waveforms_path, mmap_mode="r"), "labels": np.load(labels_path, mmap_mode="r")}
        else:
            loaded = load_pvc_windows(
                self.data_root / SOURCE_BY_KEY[record.source].directory / record.record,
                SOURCE_BY_KEY[record.source],
                lead_count=1,
            )
            if waveforms_path is not None:
                waveforms_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(waveforms_path, loaded["waveforms"])
                np.save(labels_path, loaded["labels"])
        self.cache[key] = loaded
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return loaded

    def sample_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        windows, labels, provenance = [], [], []
        eligible_sources = [
            source for source in self.sources if self.by_source_class[source][0] and self.by_source_class[source][1]
        ]
        source = eligible_sources[int(self.rng.integers(len(eligible_sources)))]
        class_counts = {0: batch_size // 2, 1: batch_size - batch_size // 2}
        for label, count in class_counts.items():
            candidates = self.by_source_class[source][label]
            record = candidates[int(self.rng.integers(len(candidates)))]
            loaded = self._load(record)
            positions = np.flatnonzero(loaded["labels"] == label)
            if not len(positions):
                raise RuntimeError(f"Manifest and loaded labels disagree for {record.key}, class {label}")
            selected = positions[self.rng.integers(len(positions), size=count)]
            for position in selected:
                windows.append(loaded["waveforms"][int(position)])
                labels.append(label)
                provenance.append({"source": source, "record": record.record, "label": label})
        order = self.rng.permutation(batch_size)
        windows = [windows[index] for index in order]
        labels = [labels[index] for index in order]
        provenance = [provenance[index] for index in order]
        return np.stack(windows), np.asarray(labels, dtype=np.int32), provenance


def validate_sampler(manifest_path: Path, data_root: Path, condition: str, batch_size: int, seed: int) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition}")
    selected_sources = set(CONDITIONS[condition])
    records = [record for record in load_training_records(manifest_path) if record.source in selected_sources]
    sampler = HierarchicalWindowSampler(records, data_root, seed)
    windows, labels, provenance = sampler.sample_batch(batch_size)
    source_counts: dict[str, int] = defaultdict(int)
    record_counts: dict[str, int] = defaultdict(int)
    for item in provenance:
        source_counts[item["source"]] += 1
        record_counts[f"{item['source']}:{item['record']}"] += 1
    return {
        "experiment": "experiment_1_sampler_smoke_test",
        "status": "passed",
        "condition": condition,
        "seed": seed,
        "batch_size": batch_size,
        "waveform_shape": list(windows.shape),
        "label_counts": {"N": int((labels == 0).sum()), "V": int((labels == 1).sum())},
        "source_counts": dict(sorted(source_counts.items())),
        "unique_records_sampled": len(record_counts),
        "forbidden_sources_observed": sorted(set(source_counts) & FORBIDDEN_SOURCES),
        "cache_limit_records": sampler.cache_size,
    }


def build_window_cache(manifest_path: Path, data_root: Path, cache_dir: Path, condition: str) -> dict:
    records = [record for record in load_training_records(manifest_path) if record.source in CONDITIONS[condition]]
    sampler = HierarchicalWindowSampler(records, data_root, seed=0, window_cache=cache_dir)
    for record in records:
        sampler._load(record)
    return {"experiment": "experiment_1_window_cache", "condition": condition, "record_count": len(records), "cache_dir": str(cache_dir), "status": "complete"}


def build_waveform_model() -> tf.keras.Model:
    """The fixed waveform-only architecture used for the Experiment 1 smoke run."""
    inputs = tf.keras.Input(shape=(300, 1), name="ecg_beat")
    x = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="class")(x)
    model = tf.keras.Model(inputs, outputs, name="experiment1_waveform_smoke")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.Recall(class_id=1, name="pvc_recall")],
    )
    return model


def smoke_train(
    manifest_path: Path,
    data_root: Path,
    splits_path: Path,
    condition: str,
    outer_fold: int,
    batch_size: int,
    steps_per_epoch: int,
    seed: int,
) -> dict:
    """Verify one outer-fold train/test path without creating official metrics.

    This deliberately does not fit calibration or select a decision threshold.
    Those operations belong to the later nested training runner.
    """
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    folds = payload["conditions"][condition]["outer_folds"]
    if not 1 <= outer_fold <= len(folds):
        raise ValueError(f"outer_fold must be in 1..{len(folds)}")
    test_keys = set(folds[outer_fold - 1]["records"])
    selected_sources = set(CONDITIONS[condition])
    all_records = [record for record in load_training_records(manifest_path) if record.source in selected_sources]
    train_records = [record for record in all_records if record.key not in test_keys]
    test_records = [record for record in all_records if record.key in test_keys]
    if not train_records or not test_records:
        raise AssertionError("Outer-fold train/test split is empty")
    if {record.key for record in train_records} & {record.key for record in test_records}:
        raise AssertionError("Outer-fold record leakage detected")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    sampler = HierarchicalWindowSampler(train_records, data_root, seed)
    model = build_waveform_model()
    losses = []
    for _ in range(steps_per_epoch):
        windows, labels, _ = sampler.sample_batch(batch_size)
        metrics = model.train_on_batch(windows, labels, return_dict=True)
        losses.append({key: float(value) for key, value in metrics.items()})

    test_sampler = HierarchicalWindowSampler(test_records, data_root, seed + 1)
    test_windows, test_labels, test_provenance = test_sampler.sample_batch(batch_size)
    evaluation = model.test_on_batch(test_windows, test_labels, return_dict=True)
    return {
        "experiment": "experiment_1_training_smoke_test",
        "status": "passed",
        "official_result": False,
        "reason": "No inner cross-fitting, calibration, or threshold selection was performed.",
        "condition": condition,
        "outer_fold": outer_fold,
        "seed": seed,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "train_record_count": len(train_records),
        "test_record_count": len(test_records),
        "train_sources": sorted({record.source for record in train_records}),
        "test_sources": sorted({record.source for record in test_records}),
        "train_metrics_last_step": losses[-1],
        "test_batch_metrics": {key: float(value) for key, value in evaluation.items()},
        "test_batch_label_counts": {"N": int((test_labels == 0).sum()), "V": int((test_labels == 1).sum())},
        "test_batch_sources": sorted({item["source"] for item in test_provenance}),
        "parameter_count": int(model.count_params()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/experiment0/record_manifest.csv"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Project root; SourceSpec directories already begin with data/",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiment1"))
    parser.add_argument("--write-splits", action="store_true")
    parser.add_argument("--validate-sampler", choices=tuple(CONDITIONS))
    parser.add_argument("--smoke-train", choices=tuple(CONDITIONS))
    parser.add_argument("--outer-fold", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=4)
    parser.add_argument("--build-window-cache", choices=tuple(CONDITIONS))
    parser.add_argument("--window-cache", type=Path, default=Path("results/experiment1/window_cache"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    if not args.write_splits and args.validate_sampler is None and args.smoke_train is None and args.build_window_cache is None:
        parser.error("Choose --write-splits, --validate-sampler CONDITION, --smoke-train CONDITION, or a combination")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_splits:
        payload = build_split_payload(args.manifest)
        path = args.output_dir / "splits.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path} with SHA-256 {payload['split_sha256']}")
    if args.validate_sampler:
        result = validate_sampler(
            args.manifest, args.data_root, args.validate_sampler, args.batch_size, args.seed
        )
        path = args.output_dir / f"sampler_smoke_{args.validate_sampler}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    if args.smoke_train:
        result = smoke_train(
            args.manifest,
            args.data_root,
            args.output_dir / "splits.json",
            args.smoke_train,
            args.outer_fold,
            args.batch_size,
            args.steps_per_epoch,
            args.seed,
        )
        path = args.output_dir / f"training_smoke_{args.smoke_train}_fold{args.outer_fold}_seed{args.seed}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    if args.build_window_cache:
        result = build_window_cache(args.manifest, args.data_root, args.window_cache, args.build_window_cache)
        path = args.output_dir / f"window_cache_{args.build_window_cache}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
