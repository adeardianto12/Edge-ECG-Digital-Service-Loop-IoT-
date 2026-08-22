"""Summarize final P0/O1 M0/M1/M2 and L1/L2 revalidation evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np


SEEDS = (20260803, 20260804, 20260805)
FOLDS = range(1, 6)
CONDITIONS = ("M0_L2", "M1_L2", "M2_L2", "M1_L1")
SPLIT_SHA256 = "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787"
BOOTSTRAP_SEED = 20260810


def aggregate(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def load_run(path: Path, condition: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment") != "final_architecture_m0_m1_m2_l1_l2_revalidation"
        or payload.get("candidate") != "P0/O1"
        or payload.get("condition") != condition
        or payload.get("split_sha256") != SPLIT_SHA256
        or payload.get("external_data_accessed")
    ):
        raise ValueError(f"Provenance validation failed: {path}")
    return payload


def metric_values(run: dict) -> dict[str, float]:
    macro = run["outer_test"]["record_macro"]
    support = macro["support_aware"]
    return {
        "support_aware_precision": float(support["pvc_precision"]),
        "support_aware_recall": float(support["pvc_recall"]),
        "support_aware_f1": float(support["pvc_f1"]),
        "legacy_zero_filled_macro_f1": float(macro["legacy_zero_filled_macro_pvc_f1"]),
        "record_macro_auprc": float(macro["record_macro_auprc"]),
        "record_macro_brier": float(macro["record_macro_brier_score"]),
        "pvc_free_false_decisions": float(macro["pvc_free_records"]["false_pvc_decisions"]),
    }


def bootstrap(deltas: np.ndarray, count: int) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(count)
    for index in range(count):
        samples[index] = np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def paired_comparison(runs: dict[str, dict[tuple[int, int], dict]], reference: str, candidate: str, resamples: int) -> dict:
    deltas, by_seed = [], []
    for seed in SEEDS:
        reference_records, candidate_records = {}, {}
        for fold in FOLDS:
            reference_records.update(runs[reference][(fold, seed)]["per_record"])
            candidate_records.update(runs[candidate][(fold, seed)]["per_record"])
        keys = sorted(set(reference_records) & set(candidate_records))
        pvc_keys = [key for key in keys if reference_records[key]["has_pvc_reference"] and candidate_records[key]["has_pvc_reference"]]
        seed_deltas = [candidate_records[key]["pvc_f1"] - reference_records[key]["pvc_f1"] for key in pvc_keys]
        deltas.extend(seed_deltas)
        by_seed.append({
            "seed": seed,
            "paired_record_count": len(keys),
            "pvc_record_count": len(pvc_keys),
            "mean_f1_difference": float(mean(seed_deltas)) if seed_deltas else None,
        })
    values = np.asarray(deltas, dtype=float)
    if not len(values):
        raise ValueError(f"No paired PVC-bearing records for {candidate} versus {reference}")
    return {
        "comparison": f"{candidate}_minus_{reference}",
        "unit": "paired outer-test PVC-bearing record",
        "mean_f1_difference": float(np.mean(values)),
        "bootstrap": {"resamples": resamples, "seed": BOOTSTRAP_SEED, "ci_95": bootstrap(values, resamples)},
        "seed_results": by_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_revalidation"))
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1_000:
        parser.error("At least 1,000 record bootstrap resamples are required")

    runs: dict[str, dict[tuple[int, int], dict]] = {condition: {} for condition in CONDITIONS}
    rows, grouped = [], defaultdict(list)
    for condition in CONDITIONS:
        for fold in FOLDS:
            for seed in SEEDS:
                path = args.output_dir / f"{condition}_fold{fold}_seed{seed}.json"
                if not path.exists():
                    raise FileNotFoundError(f"Missing required revalidation evidence: {path}")
                run = load_run(path, condition)
                runs[condition][(fold, seed)] = run
                values = metric_values(run)
                row = {
                    "condition": condition,
                    "source_condition": run["source_condition"],
                    "lead_profile": run["lead_profile"],
                    "fold": fold,
                    "seed": seed,
                    "parameter_count": run["architecture"]["parameter_count"],
                    "threshold": run["selection"]["threshold"],
                    **values,
                }
                rows.append(row)
                grouped[condition].append(row)

    fieldnames = list(rows[0])
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    conditions = {}
    for condition, condition_rows in grouped.items():
        metric_names = [key for key in condition_rows[0] if key.startswith(("support_", "legacy_", "record_", "pvc_free_"))]
        seed_means = {
            str(seed): {
                metric: float(mean(row[metric] for row in condition_rows if row["seed"] == seed))
                for metric in metric_names
            }
            for seed in SEEDS
        }
        conditions[condition] = {
            "run_count": len(condition_rows),
            "expected_run_count": 15,
            "source_condition": condition_rows[0]["source_condition"],
            "lead_profile": condition_rows[0]["lead_profile"],
            "parameter_count": sorted({row["parameter_count"] for row in condition_rows}),
            "all_runs": {metric: aggregate([row[metric] for row in condition_rows]) for metric in metric_names},
            "seed_means": seed_means,
            "three_seed": {metric: aggregate([seed_means[str(seed)][metric] for seed in SEEDS]) for metric in metric_names},
        }

    comparisons = [
        paired_comparison(runs, "M0_L2", "M1_L2", args.bootstrap_resamples),
        paired_comparison(runs, "M1_L2", "M2_L2", args.bootstrap_resamples),
        paired_comparison(runs, "M1_L1", "M1_L2", args.bootstrap_resamples),
    ]
    summary = {
        "experiment": "final_architecture_m0_m1_m2_l1_l2_revalidation",
        "status": "complete",
        "candidate": "P0/O1",
        "split_sha256": SPLIT_SHA256,
        "conditions": conditions,
        "paired_record_comparisons": comparisons,
        "selection_rule": "Report paired development evidence and resource cost; do not select using external data or best seed.",
        "external_data_accessed": False,
        "prohibited_sources": ["nsrdb", "svdb"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
