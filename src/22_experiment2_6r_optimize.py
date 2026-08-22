"""Run one preregistered Experiment 2.6R M1/L2 outer-fold candidate.

The runner is isolated from Experiment 2.6 evidence and refuses every source
except MIT-BIH and INCART.  R1 stabilizes training with EMA, R2 uses a
development-only cross-fitted hard-PVC cache, and R3 changes only the
inner-fold decision rule.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

SPLIT_SHA256 = "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787"
SOURCES = {"mitdb", "incartdb"}
FORBIDDEN_SOURCES = {"svdb", "nsrdb"}
CANDIDATES = ("R1", "R2", "R3")
P0_STAGE_A_PVC_FREE_FP = 161
EMA_DECAY = 0.999
R2_HARD_QUANTILE = 0.25
R2_HARD_WEIGHT = 1.15
R2_PROXY_EPOCHS = 10


def load_core():
    path = Path(__file__).with_name("14_experiment2_5_optimize.py")
    spec = importlib.util.spec_from_file_location("experiment2_6r_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Experiment 2.5 core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load_core()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EMA:
    def __init__(self, model, decay: float) -> None:
        self.decay = decay
        self.weights = [np.array(weight, copy=True) for weight in model.get_weights()]

    def update(self, model) -> None:
        self.weights = [
            self.decay * averaged + (1.0 - self.decay) * current
            for averaged, current in zip(self.weights, model.get_weights())
        ]

    def apply(self, model) -> list[np.ndarray]:
        original = model.get_weights()
        model.set_weights(self.weights)
        return original


class HardPVCSampler(core.HierarchicalSampler):
    def __init__(self, records, store, seed: int, hard_positions: dict[str, np.ndarray] | None) -> None:
        super().__init__(records, store, seed, augment=False)
        self.hard_positions = hard_positions or {}

    def sample(self, batch_size: int):
        eligible = [source for source, values in self.by_source_class.items() if values[0] and values[1]]
        source = eligible[int(self.rng.integers(len(eligible)))]
        windows, rr, labels = [], [], []
        for label, count in ((0, batch_size // 2), (1, batch_size - batch_size // 2)):
            record = self.by_source_class[source][label][int(self.rng.integers(len(self.by_source_class[source][label])))]
            loaded = self.store.load(record)
            positions = np.flatnonzero(loaded["labels"] == label)
            if label == 1 and record.key in self.hard_positions:
                hard = self.hard_positions[record.key][positions]
                probabilities = np.where(hard, R2_HARD_WEIGHT, 1.0).astype(np.float64)
                probabilities /= probabilities.sum()
                choice = self.rng.choice(positions, size=count, replace=True, p=probabilities)
            else:
                choice = positions[self.rng.integers(len(positions), size=count)]
            windows.extend(loaded["waveforms"][int(item)] for item in choice)
            rr.extend(loaded["rr"][int(item)] for item in choice)
            labels.extend([label] * count)
        order = self.rng.permutation(batch_size)
        return (
            np.stack([windows[item] for item in order]).astype(np.float32),
            np.stack([rr[item] for item in order]).astype(np.float32),
            np.asarray([labels[item] for item in order], dtype=np.int32),
        )


def train_model(records, store, args, seed: int, epochs: int, candidate: str, hard_positions=None, use_ema=False):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    sampler = HardPVCSampler(records, store, seed, hard_positions if candidate == "R2" else None)
    model = core.build_model("O1")
    ema = EMA(model, EMA_DECAY) if use_ema else None
    for _ in range(epochs):
        for _ in range(args.steps_per_epoch):
            waveforms, rr, labels = sampler.sample(args.batch_size)
            model.train_on_batch(core.model_inputs("O1", waveforms, rr), labels)
            if ema is not None:
                ema.update(model)
    if ema is not None:
        ema.apply(model)
    return model


def train_select_epoch(records, validation_records, store, args, seed, candidate, hard_positions):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    sampler = HardPVCSampler(records, store, seed, hard_positions if candidate == "R2" else None)
    model = core.build_model("O1")
    ema = EMA(model, EMA_DECAY) if candidate == "R1" else None
    curve, best_epoch, best_score, best_weights = [], 1, -np.inf, None
    for epoch in range(1, args.epochs + 1):
        for _ in range(args.steps_per_epoch):
            waveforms, rr, labels = sampler.sample(args.batch_size)
            model.train_on_batch(core.model_inputs("O1", waveforms, rr), labels)
            if ema is not None:
                ema.update(model)
        original = ema.apply(model) if ema is not None else None
        predictions = core.predict_records(model, "O1", validation_records, store)
        if original is not None:
            model.set_weights(original)
        _, metrics = core.aggregate_records(predictions, 0.5)
        score = metrics["record_macro"]["record_macro_auprc"]
        curve.append({"epoch": epoch, "selection_record_macro_auprc": score})
        if score is not None and score > best_score:
            best_epoch, best_score = epoch, score
            best_weights = (ema.weights if ema is not None else model.get_weights())
    if best_weights is None:
        raise RuntimeError("Inner epoch selection did not yield a valid model")
    model.set_weights(best_weights)
    return model, best_epoch, curve


def build_r2_hard_positions(development, inner_folds, store, args, seed: int, output_dir: Path) -> dict[str, np.ndarray]:
    """Cross-fit raw O1 probabilities using development records only."""
    proxy_predictions = {}
    for index, fold in enumerate(inner_folds):
        validation_keys = set(fold["records"])
        train_records = [record for record in development if record.key not in validation_keys]
        validation_records = [record for record in development if record.key in validation_keys]
        proxy = train_model(train_records, store, args, seed + 1000 + index, R2_PROXY_EPOCHS, "R2")
        proxy_predictions.update(core.predict_records(proxy, "O1", validation_records, store))
    all_scores = {source: [] for source in SOURCES}
    for key, (labels, probabilities) in proxy_predictions.items():
        source = key.split(":", 1)[0]
        for position in np.flatnonzero(labels == 1):
            all_scores[source].append((key, int(position), float(probabilities[position])))
    hard_positions = {}
    source_thresholds = {}
    for source, values in all_scores.items():
        if not values:
            raise RuntimeError(f"R2 proxy cache has no PVC examples for {source}")
        cutoff = float(np.quantile([item[2] for item in values], R2_HARD_QUANTILE))
        source_thresholds[source] = cutoff
        for key, position, probability in values:
            if probability <= cutoff:
                if key not in hard_positions:
                    hard_positions[key] = np.zeros(len(store.load(next(record for record in development if record.key == key))["labels"]), dtype=bool)
                hard_positions[key][position] = True
    cache_dir = output_dir / "hard_pvc_proxy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"R2_outer_proxy_seed{seed}.json"
    cache_file.write_text(json.dumps({
        "outer_development_only": True,
        "proxy_candidate": "P0/O1",
        "proxy_epochs": R2_PROXY_EPOCHS,
        "cross_fit_folds": len(inner_folds),
        "hard_quantile": R2_HARD_QUANTILE,
        "hard_weight": R2_HARD_WEIGHT,
        "source_probability_cutoffs": source_thresholds,
        "hard_pvc_counts": {key: int(np.sum(value)) for key, value in hard_positions.items()},
    }, indent=2) + "\n", encoding="utf-8")
    return hard_positions


def calibrate(predictions):
    labels = np.concatenate([values[0] for values in predictions.values()])
    raw = np.concatenate([values[1] for values in predictions.values()])
    logits = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
    scaler = LogisticRegression(random_state=0, max_iter=1000).fit(
        logits.reshape(-1, 1), labels, sample_weight=core.balanced_sample_weights(predictions)
    )
    calibrated, cursor = {}, 0
    transformed = scaler.predict_proba(logits.reshape(-1, 1))[:, 1]
    for key, (record_labels, _) in predictions.items():
        calibrated[key] = (record_labels, transformed[cursor:cursor + len(record_labels)])
        cursor += len(record_labels)
    return scaler, calibrated


def choose_standard(calibrated):
    options = []
    for threshold in np.arange(0.01, 1.00, 0.01):
        _, summary = core.aggregate_records(calibrated, float(threshold))
        support = summary["record_macro"]["support_aware"]
        options.append({"threshold": float(threshold), "pooled_recall": summary["pooled"]["pvc_recall"],
                        "support_aware_recall": support["pvc_recall"], "support_aware_f1": support["pvc_f1"],
                        "pvc_free_false_decisions": summary["record_macro"]["pvc_free_records"]["false_pvc_decisions"]})
    feasible = [item for item in options if item["pooled_recall"] >= .90 and item["support_aware_recall"] >= .90]
    selected = dict(max(feasible or options, key=lambda item: (
        item["support_aware_f1"], item["pooled_recall"], -item["pvc_free_false_decisions"], -item["threshold"])))
    selected["recall_target_met"] = bool(feasible)
    return selected, options


def select_r3(calibrated_by_inner):
    combined = {key: value for predictions in calibrated_by_inner.values() for key, value in predictions.items()}
    baseline, _ = choose_standard(combined)
    options = []
    for threshold in np.arange(.01, 1.00, .01):
        inner_summaries = [core.aggregate_records(predictions, float(threshold))[1] for predictions in calibrated_by_inner.values()]
        recalls = [summary["record_macro"]["support_aware"]["pvc_recall"] for summary in inner_summaries]
        total_fp = sum(summary["record_macro"]["pvc_free_records"]["false_pvc_decisions"] for summary in inner_summaries)
        f1 = float(np.mean([summary["record_macro"]["support_aware"]["pvc_f1"] for summary in inner_summaries]))
        options.append({"threshold": float(threshold), "minimum_inner_recall": min(recalls),
                        "mean_inner_support_aware_f1": f1, "inner_pvc_free_false_decisions": int(total_fp)})
    feasible = [item for item in options if item["minimum_inner_recall"] >= .90
                and item["inner_pvc_free_false_decisions"] <= baseline["pvc_free_false_decisions"]]
    if not feasible:
        return baseline["threshold"], {"r3_feasible": False, "fallback": "standard_P0_rule",
                                       "p0_inner_fp_budget": baseline["pvc_free_false_decisions"],
                                       "threshold_trace": options}, baseline
    selected = max(feasible, key=lambda item: (item["minimum_inner_recall"], item["mean_inner_support_aware_f1"],
                                                -item["inner_pvc_free_false_decisions"], -item["threshold"]))
    return selected["threshold"], {"r3_feasible": True, "p0_inner_fp_budget": baseline["pvc_free_false_decisions"],
                                   "threshold_trace": options, **selected}, baseline


def apply_scaler(scaler, probabilities):
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))
    return scaler.predict_proba(logits.reshape(-1, 1))[:, 1]


def write_protocol(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "experiment_2_6r_recall_stability_revision",
        "candidates": {"R1": "P0/O1 plus EMA weight averaging", "R2": "P0/O1 plus cross-fitted hard-PVC sampling",
                       "R3": "P0/O1 plus per-inner-fold recall-safe decision selection"},
        "sources": sorted(SOURCES), "forbidden_sources": sorted(FORBIDDEN_SOURCES), "split_sha256": SPLIT_SHA256,
        "stage_a_seed": 20260803, "stage_b_seeds": [20260803, 20260804, 20260805],
        "p0_stage_a_pvc_free_fp": P0_STAGE_A_PVC_FREE_FP, "external_data_accessed": False,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


def main():
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
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiment2_6r"))
    parser.add_argument("--window-cache", type=Path, default=Path("results/experiment2_6r/window_cache"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 20 <= args.epochs <= 30 or args.steps_per_epoch < 1 or args.batch_size < 2:
        parser.error("Invalid preregistered training budget")
    payload = json.loads(args.splits.read_text(encoding="utf-8"))
    if payload.get("split_sha256") != SPLIT_SHA256:
        raise ValueError("Unexpected split hash")
    condition = payload["conditions"].get("M1")
    if condition is None or set(condition["sources"]) != SOURCES:
        raise ValueError("2.6R is restricted to MIT-BIH plus INCART")
    if not 1 <= args.outer_fold <= len(condition["outer_folds"]):
        parser.error("outer fold outside frozen split")
    records = [record for record in core.prep.load_training_records(args.manifest) if record.source in SOURCES]
    if {record.source for record in records} != SOURCES or any(record.source in FORBIDDEN_SOURCES for record in records):
        raise ValueError("2.6R source guard failed")
    test_keys = set(condition["outer_folds"][args.outer_fold - 1]["records"])
    development = [record for record in records if record.key not in test_keys]
    test = [record for record in records if record.key in test_keys]
    if {record.key for record in development} & {record.key for record in test}:
        raise AssertionError("Outer-test record leakage")
    write_protocol(args.output_dir)
    if args.dry_run:
        print(json.dumps({"status": "passed", "sources": sorted(SOURCES), "external_data_accessed": False}, indent=2))
        return
    store = core.RecordStore(records, args.data_root, args.window_cache)
    inner_folds = [fold for index, fold in enumerate(condition["outer_folds"]) if index != args.outer_fold - 1]
    hard_positions = build_r2_hard_positions(development, inner_folds, store, args, args.seed, args.output_dir) if args.candidate == "R2" else None
    oof_by_inner, curves, epochs = {}, {}, []
    for index, fold in enumerate(inner_folds):
        validation_keys = set(fold["records"])
        inner_train = [record for record in development if record.key not in validation_keys]
        inner_validation = [record for record in development if record.key in validation_keys]
        model, selected_epoch, curve = train_select_epoch(inner_train, inner_validation, store, args, args.seed + index,
                                                            args.candidate, hard_positions)
        oof_by_inner[f"inner_{index + 1}"] = core.predict_records(model, "O1", inner_validation, store)
        curves[f"inner_{index + 1}"] = curve
        epochs.append(selected_epoch)
    oof = {key: value for predictions in oof_by_inner.values() for key, value in predictions.items()}
    scaler, calibrated_oof = calibrate(oof)
    if args.candidate == "R3":
        calibrated_by_inner, cursor = {}, 0
        for name, predictions in oof_by_inner.items():
            calibrated_by_inner[name] = {}
            for key, (labels, raw) in predictions.items():
                calibrated_by_inner[name][key] = (labels, calibrated_oof[key][1])
                cursor += len(labels)
        threshold, selection, baseline = select_r3(calibrated_by_inner)
        selection["calibration"] = "P0 uncropped Platt scaling with per-inner-fold safety constraint"
        selection["baseline_selection"] = baseline
    else:
        selection, trace = choose_standard(calibrated_oof)
        threshold = selection["threshold"]
        selection["calibration"] = "P0 uncropped Platt scaling with source- and record-balanced weights"
        selection["threshold_trace"] = trace
    final_epochs = int(np.median(epochs))
    final_model = train_model(development, store, args, args.seed + 100, final_epochs, args.candidate, hard_positions,
                              use_ema=args.candidate == "R1")
    test_raw = core.predict_records(final_model, "O1", test, store)
    test_calibrated = {key: (labels, apply_scaler(scaler, values)) for key, (labels, values) in test_raw.items()}
    per_record, outer_test = core.aggregate_records(test_calibrated, threshold)
    prediction_dir = args.output_dir / "per_record_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for key, (labels, probabilities) in test_calibrated.items():
        np.savez_compressed(prediction_dir / f"{args.candidate}_fold{args.outer_fold}_seed{args.seed}_{key.replace(':', '_')}.npz",
                            labels=labels, calibrated_probability=probabilities)
    curve_dir = args.output_dir / "learning_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    (curve_dir / f"{args.candidate}_fold{args.outer_fold}_seed{args.seed}.json").write_text(json.dumps(curves, indent=2) + "\n")
    architecture = final_model.to_json()
    result = {"experiment": "experiment_2_6r_recall_stability_revision",
              "stage": "A" if args.seed == 20260803 else "B", "candidate": args.candidate,
              "outer_fold": args.outer_fold, "seed": args.seed, "split_sha256": SPLIT_SHA256,
              "sources": sorted(SOURCES), "lead_profile": "L2", "input_shape": [300, 2],
              "external_data_accessed": False,
              "training": {"max_epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size,
                           "final_epochs": final_epochs, "inner_selected_epochs": epochs,
                           "sampler": "hierarchical source -> record -> class"},
              "selection": selection | {"threshold": threshold, "epoch_selection_metric": "inner record-macro AUPRC"},
              "architecture": {"parameter_count": int(final_model.count_params()),
                               "json_sha256": hashlib.sha256(architecture.encode("utf-8")).hexdigest()},
              "outer_test": outer_test, "per_record": per_record, "source_code_sha256": sha256_file(Path(__file__))}
    path = args.output_dir / f"{args.candidate}_fold{args.outer_fold}_seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

