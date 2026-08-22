"""Create auditable Experiment 1 run and condition summaries.

This utility reads only Experiment 1 development outputs.  It never opens
database files, so it cannot access SVDB or NSRDB.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


CONDITIONS = ("M0", "M1", "M2")
FOLDS = range(1, 6)
SEEDS = (20260803, 20260804, 20260805)
EXPECTED_SPLIT_SHA256 = "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787"
METRICS = ("pvc_precision", "pvc_recall", "pvc_f1", "specificity", "balanced_accuracy", "auroc", "auprc", "brier_score")


def aggregate(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def record_macro(per_record: dict[str, dict]) -> dict[str, float | None]:
    output = {}
    for metric in METRICS:
        values = [item[metric] for item in per_record.values() if item[metric] is not None]
        output[metric] = mean(values) if values else None
    return output


def main() -> None:
    results_dir = Path("results/experiment1")
    split = json.loads((results_dir / "splits.json").read_text(encoding="utf-8"))
    if split["split_sha256"] != EXPECTED_SPLIT_SHA256:
        raise ValueError("Unexpected frozen split hash")

    rows = []
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for condition in CONDITIONS:
        for fold in FOLDS:
            for seed in SEEDS:
                path = results_dir / f"{condition}_fold{fold}_seed{seed}.json"
                if not path.exists():
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                if not result.get("official_result") or result.get("status") != "complete":
                    raise ValueError(f"Non-official or incomplete result: {path}")
                if result.get("split_sha256") != EXPECTED_SPLIT_SHA256:
                    raise ValueError(f"Split hash mismatch: {path}")
                macro = record_macro(result["per_record"])
                row = {
                    "condition": condition,
                    "outer_fold": fold,
                    "seed": seed,
                    "threshold": result["selection"]["threshold"],
                    "selection_recall_target_met": result["selection"]["recall_target_met"],
                    "selection_record_macro_pvc_f1": result["selection"]["record_macro_pvc_f1"],
                    "selection_pvc_recall": result["selection"]["pvc_recall"],
                    "outer_record_count": len(result["per_record"]),
                }
                for metric in METRICS:
                    row[f"outer_pooled_{metric}"] = result["outer_test"][metric]
                    row[f"outer_record_macro_{metric}"] = macro[metric]
                rows.append(row)
                by_condition[condition].append(row)

    fieldnames = list(rows[0]) if rows else ["condition", "outer_fold", "seed"]
    with (results_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    conditions = {}
    for condition in CONDITIONS:
        condition_rows = by_condition[condition]
        by_seed = {}
        for seed in SEEDS:
            seed_rows = [row for row in condition_rows if row["seed"] == seed]
            by_seed[str(seed)] = {
                "run_count": len(seed_rows),
                "record_macro": {
                    metric: aggregate([row[f"outer_record_macro_{metric}"] for row in seed_rows if row[f"outer_record_macro_{metric}"] is not None])
                    for metric in METRICS
                },
            }
        conditions[condition] = {
            "status": "complete" if len(condition_rows) == 15 else "pending" if not condition_rows else "incomplete",
            "run_count": len(condition_rows),
            "expected_run_count": 15,
            "record_macro": {
                metric: aggregate([row[f"outer_record_macro_{metric}"] for row in condition_rows if row[f"outer_record_macro_{metric}"] is not None])
                for metric in METRICS
            },
            "pooled": {
                metric: aggregate([row[f"outer_pooled_{metric}"] for row in condition_rows if row[f"outer_pooled_{metric}"] is not None])
                for metric in METRICS
            },
            "by_seed": by_seed,
        }

    summary = {
        "experiment": "experiment_1_multisource_training_ablation",
        "status": "complete" if all(item["status"] == "complete" for item in conditions.values()) else "in_progress",
        "official_result": True,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "selection_rule": "development only; require PVC recall >= 0.90 if feasible, then maximize record-level macro PVC F1",
        "conditions": conditions,
        "external_data_accessed": False,
        "interpretation": "Development-only summary. It cannot select a final model until M2 completes and must not be used to tune SVDB or NSRDB.",
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {results_dir / 'runs.csv'} and {results_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
