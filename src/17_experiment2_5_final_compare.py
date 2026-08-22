"""Paired record-bootstrap comparison for confirmed Experiment 2.5 candidates."""

import argparse
import json
from pathlib import Path

import numpy as np


SEEDS = (20260803, 20260804, 20260805)
FOLDS = range(1, 6)
BOOTSTRAP_SEED = 20260806


def record_metrics(path: Path, threshold: float) -> dict:
    payload = np.load(path)
    labels = payload["labels"].astype(int)
    predicted = payload["calibrated_probability"] >= threshold
    true_positive = int(np.sum(predicted & (labels == 1)))
    false_positive = int(np.sum(predicted & (labels == 0)))
    true_negative = int(np.sum(~predicted & (labels == 0)))
    false_negative = int(np.sum(~predicted & (labels == 1)))
    has_pvc = bool(np.any(labels == 1))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "has_pvc": has_pvc,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12) if has_pvc else None,
        "false_positive": false_positive,
        "specificity": true_negative / max(true_negative + false_positive, 1),
    }


def thresholds(output_dir: Path, candidate: str) -> dict[tuple[int, int], float]:
    result = {}
    for seed in SEEDS:
        for fold in FOLDS:
            payload = json.loads((output_dir / f"{candidate}_fold{fold}_seed{seed}.json").read_text(encoding="utf-8"))
            result[(seed, fold)] = float(payload["selection"]["threshold"])
    return result


def load_records(output_dir: Path, prediction_dir: Path, candidate: str) -> dict[int, dict[str, dict]]:
    candidate_thresholds = thresholds(output_dir, candidate)
    output: dict[int, dict[str, dict]] = {}
    for seed in SEEDS:
        records = {}
        for fold in FOLDS:
            prefix = f"{candidate}_fold{fold}_seed{seed}_"
            for path in prediction_dir.glob(f"{prefix}*.npz"):
                key = path.stem[len(prefix):]
                records[key] = record_metrics(path, candidate_thresholds[(seed, fold)])
        output[seed] = records
    return output


def bootstrap(deltas: np.ndarray, rng: np.random.Generator, count: int) -> tuple[float, float]:
    samples = np.empty(count)
    for index in range(count):
        samples[index] = np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiment2_5"))
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    prediction_dir = args.output_dir / "per_record_predictions"
    o1 = load_records(args.output_dir, prediction_dir, "O1")
    o2 = load_records(args.output_dir, prediction_dir, "O2")
    seed_results = []
    all_deltas = []
    for seed in SEEDS:
        keys = sorted(set(o1[seed]) & set(o2[seed]))
        if not keys:
            raise ValueError(f"No paired records for seed {seed}")
        pvc_keys = [key for key in keys if o1[seed][key]["has_pvc"] and o2[seed][key]["has_pvc"]]
        pvc_free_keys = [key for key in keys if not o1[seed][key]["has_pvc"] and not o2[seed][key]["has_pvc"]]
        f1_deltas = np.asarray([o2[seed][key]["f1"] - o1[seed][key]["f1"] for key in pvc_keys])
        all_deltas.extend(f1_deltas.tolist())
        seed_results.append({
            "seed": seed,
            "paired_record_count": len(keys),
            "pvc_record_count": len(pvc_keys),
            "pvc_free_record_count": len(pvc_free_keys),
            "mean_f1_difference_o2_minus_o1": float(np.mean(f1_deltas)),
            "o1_pvc_free_false_positive_mean": float(np.mean([o1[seed][key]["false_positive"] for key in pvc_free_keys])) if pvc_free_keys else None,
            "o2_pvc_free_false_positive_mean": float(np.mean([o2[seed][key]["false_positive"] for key in pvc_free_keys])) if pvc_free_keys else None,
            "o1_pvc_free_specificity_mean": float(np.mean([o1[seed][key]["specificity"] for key in pvc_free_keys])) if pvc_free_keys else None,
            "o2_pvc_free_specificity_mean": float(np.mean([o2[seed][key]["specificity"] for key in pvc_free_keys])) if pvc_free_keys else None,
        })
    deltas = np.asarray(all_deltas)
    lower, upper = bootstrap(deltas, np.random.default_rng(BOOTSTRAP_SEED), args.bootstrap_resamples)
    summary = {
        "experiment": "experiment_2_5_pre_freeze_model_optimization",
        "comparison": "O2_minus_O1",
        "unit": "paired outer-test record with PVC reference",
        "bootstrap": {"resamples": args.bootstrap_resamples, "seed": BOOTSTRAP_SEED, "ci_95": [lower, upper]},
        "mean_f1_difference_o2_minus_o1": float(np.mean(deltas)),
        "lower_ci_above_zero": lower > 0,
        "seed_results": seed_results,
        "source_boundary": {"sources": ["incartdb", "mitdb"], "external_data_accessed": False, "prohibited_sources": ["nsrdb", "svdb"]},
    }
    path = args.output_dir / "final_candidate_comparison.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(path), "lower_ci_above_zero": summary["lower_ci_above_zero"]}, indent=2))


if __name__ == "__main__":
    main()
