"""Run one Experiment 1 outer fold with nested record-group calibration.

Run only after reviewing results/experiment1/splits.json.  SVDB and NSRDB are
not represented in this program and cannot be passed as conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from multisource_ecg import SOURCE_BY_KEY, load_pvc_windows
from importlib.util import module_from_spec, spec_from_file_location


PREP_PATH = Path(__file__).with_name("08_experiment1_prepare.py")
SPEC = spec_from_file_location("experiment1_prepare", PREP_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load Experiment 1 preparation module")
prep = module_from_spec(SPEC)
sys.modules[SPEC.name] = prep
SPEC.loader.exec_module(prep)


def selected_records(manifest: Path, sources: set[str], keys: set[str]) -> list:
    records = [item for item in prep.load_training_records(manifest) if item.source in sources]
    result = [item for item in records if item.key in keys]
    if {item.key for item in result} != keys:
        raise ValueError("Split manifest contains a record absent from the audited training manifest")
    return result


def train(
    records: list,
    data_root: Path,
    seed: int,
    epochs: int,
    steps: int,
    batch_size: int,
    window_cache: Path,
):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    cache_size = len(records) if {record.source for record in records} == {"mitdb"} else 8
    sampler = prep.HierarchicalWindowSampler(
        records,
        data_root,
        seed,
        cache_size=cache_size,
        window_cache=window_cache,
    )
    model = prep.build_waveform_model()
    for _ in range(epochs):
        for _ in range(steps):
            x, y, _ = sampler.sample_batch(batch_size)
            model.train_on_batch(x, y)
    return model


def load_record_windows(item, data_root: Path, window_cache: Path) -> dict:
    """Read the audited cached array when present, without changing its contents."""
    record_cache = window_cache / item.source
    waveforms_path = record_cache / f"{item.record}_waveforms.npy"
    labels_path = record_cache / f"{item.record}_labels.npy"
    if waveforms_path.exists() and labels_path.exists():
        return {
            "waveforms": np.load(waveforms_path, mmap_mode="r"),
            "labels": np.load(labels_path, mmap_mode="r"),
        }
    return load_pvc_windows(
        data_root / SOURCE_BY_KEY[item.source].directory / item.record,
        SOURCE_BY_KEY[item.source],
        lead_count=1,
    )


def predict_records(
    model, records: list, data_root: Path, window_cache: Path
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for item in records:
        loaded = load_record_windows(item, data_root, window_cache)
        if not len(loaded["labels"]):
            continue
        probabilities = model.predict(loaded["waveforms"], batch_size=1024, verbose=0)[:, 1]
        output[item.key] = (loaded["labels"], probabilities)
    return output


def calibrate(raw: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[LogisticRegression, float, dict]:
    labels = np.concatenate([value[0] for value in raw.values()])
    probabilities = np.concatenate([value[1] for value in raw.values()])
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))
    scaler = LogisticRegression(random_state=0).fit(logits.reshape(-1, 1), labels)
    calibrated = scaler.predict_proba(logits.reshape(-1, 1))[:, 1]
    candidates = []
    for threshold in np.arange(0.01, 1.0, 0.01):
        predictions = calibrated >= threshold
        recall = float(((predictions == 1) & (labels == 1)).sum() / max((labels == 1).sum(), 1))
        record_f1 = []
        for record_labels, record_probs in raw.values():
            record_pred = calibrated[: len(record_labels)] >= threshold
            calibrated = calibrated[len(record_labels) :]
            tp = ((record_pred == 1) & (record_labels == 1)).sum()
            fp = ((record_pred == 1) & (record_labels == 0)).sum()
            fn = ((record_pred == 0) & (record_labels == 1)).sum()
            record_f1.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
        calibrated = scaler.predict_proba(logits.reshape(-1, 1))[:, 1]
        candidates.append((recall >= 0.90, np.mean(record_f1), recall, float(threshold)))
    feasible = [item for item in candidates if item[0]]
    chosen = max(feasible or candidates, key=lambda item: (item[1], item[2], item[3]))
    return scaler, chosen[3], {"recall_target_met": chosen[0], "record_macro_pvc_f1": chosen[1], "pvc_recall": chosen[2]}


def apply_scaler(scaler: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))
    return scaler.predict_proba(logits.reshape(-1, 1))[:, 1]


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    pred = probabilities >= threshold
    tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); specificity = tn / max(tn + fp, 1)
    has_both_classes = len(np.unique(labels)) == 2
    return {"pvc_precision": precision, "pvc_recall": recall, "pvc_f1": 2 * precision * recall / max(precision + recall, 1e-12), "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2, "auroc": float(roc_auc_score(labels, probabilities)) if has_both_classes else None, "auprc": float(average_precision_score(labels, probabilities)) if has_both_classes else None, "brier_score": float(brier_score_loss(labels, probabilities)), "confusion_matrix": [[tn, fp], [fn, tp]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=tuple(prep.CONDITIONS), required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pilot", action="store_true", help="Mark a reduced-step protocol check as non-official")
    parser.add_argument("--manifest", type=Path, default=Path("results/experiment0/record_manifest.csv"))
    parser.add_argument("--splits", type=Path, default=Path("results/experiment1/splits.json"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiment1"))
    parser.add_argument("--window-cache", type=Path, default=Path("results/experiment1/window_cache"))
    args = parser.parse_args()
    payload = json.loads(args.splits.read_text(encoding="utf-8"))
    condition = payload["conditions"][args.condition]
    folds = condition["outer_folds"]
    if not 1 <= args.outer_fold <= len(folds):
        parser.error("--outer-fold is outside the frozen split manifest")
    sources = set(condition["sources"])
    test_keys = set(folds[args.outer_fold - 1]["records"])
    development_keys = set().union(*(set(fold["records"]) for index, fold in enumerate(folds) if index != args.outer_fold - 1))
    development = selected_records(args.manifest, sources, development_keys)
    test = selected_records(args.manifest, sources, test_keys)

    # Four inner record groups provide cross-fitted, out-of-fold probabilities.
    oof = {}
    for inner_index, inner_fold in enumerate(fold for index, fold in enumerate(folds) if index != args.outer_fold - 1):
        calibration_keys = set(inner_fold["records"])
        inner_train = [item for item in development if item.key not in calibration_keys]
        inner_calibration = [item for item in development if item.key in calibration_keys]
        model = train(
            inner_train,
            args.data_root,
            args.seed + inner_index,
            args.epochs,
            args.steps_per_epoch,
            args.batch_size,
            args.window_cache,
        )
        oof.update(predict_records(model, inner_calibration, args.data_root, args.window_cache))
    scaler, threshold, selection = calibrate(oof)
    final_model = train(
        development,
        args.data_root,
        args.seed + 100,
        args.epochs,
        args.steps_per_epoch,
        args.batch_size,
        args.window_cache,
    )
    test_predictions = predict_records(final_model, test, args.data_root, args.window_cache)
    per_record = {}
    for key, (labels, raw_probabilities) in test_predictions.items():
        calibrated = apply_scaler(scaler, raw_probabilities)
        per_record[key] = metrics(labels, calibrated, threshold)
        path = args.output_dir / "per_record_predictions" / f"{args.condition}_fold{args.outer_fold}_seed{args.seed}_{key.replace(':', '_')}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, labels=labels, calibrated_probability=calibrated)
    labels = np.concatenate([item[0] for item in test_predictions.values()])
    probabilities = np.concatenate([apply_scaler(scaler, item[1]) for item in test_predictions.values()])
    result = {"experiment": "experiment_1_multisource_training_ablation", "status": "pilot" if args.pilot else "complete", "official_result": not args.pilot, "condition": args.condition, "outer_fold": args.outer_fold, "seed": args.seed, "split_sha256": payload["split_sha256"], "training": {"epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size, "sources": sorted(sources), "window_cache": str(args.window_cache)}, "selection": selection | {"threshold": threshold, "calibration": "Platt scaling fitted on inner out-of-fold record predictions"}, "outer_test": metrics(labels, probabilities, threshold), "per_record": per_record}
    output = args.output_dir / f"{args.condition}_fold{args.outer_fold}_seed{args.seed}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
