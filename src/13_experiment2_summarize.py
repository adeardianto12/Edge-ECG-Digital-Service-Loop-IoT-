"""Summarize the M1 L1 baseline and new L2 Experiment 2 ablation results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


SPLIT_SHA256 = "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787"
FOLDS = range(1, 6)
SEEDS = (20260803, 20260804, 20260805)
METRICS = ("pvc_precision", "pvc_recall", "pvc_f1", "specificity", "balanced_accuracy", "auroc", "auprc", "brier_score")


def aggregate(values: list[float]) -> dict[str, float | None]:
    return {"mean": mean(values) if values else None, "std": stdev(values) if len(values) > 1 else 0.0 if values else None, "min": min(values) if values else None, "max": max(values) if values else None}


def macro(per_record: dict[str, dict]) -> dict[str, float | None]:
    return {metric: mean([item[metric] for item in per_record.values() if item[metric] is not None]) if any(item[metric] is not None for item in per_record.values()) else None for metric in METRICS}


def load_l1(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("condition") != "M1" or result.get("split_sha256") != SPLIT_SHA256 or set(result["training"]["sources"]) != {"mitdb", "incartdb"}:
        raise ValueError(f"L1 provenance check failed: {path}")
    return result


def load_l2(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("official_result") or result.get("lead_profile") != "L2" or result.get("split_sha256") != SPLIT_SHA256:
        raise ValueError(f"L2 provenance check failed: {path}")
    return result


def main() -> None:
    experiment1 = Path("results/experiment1")
    experiment2 = Path("results/experiment2")
    rows, by_profile = [], defaultdict(list)
    for profile, loader, directory, prefix in (("L1", load_l1, experiment1, "M1"), ("L2", load_l2, experiment2, "L2")):
        for fold in FOLDS:
            for seed in SEEDS:
                path = directory / f"{prefix}_fold{fold}_seed{seed}.json"
                if not path.exists():
                    continue
                result = loader(path)
                record_macro = macro(result["per_record"])
                row = {"lead_profile": profile, "outer_fold": fold, "seed": seed, "threshold": result["selection"]["threshold"], "outer_record_count": len(result["per_record"]), "parameter_count": result.get("architecture", {}).get("parameter_count")}
                for metric in METRICS:
                    row[f"outer_pooled_{metric}"] = result["outer_test"][metric]
                    row[f"outer_record_macro_{metric}"] = record_macro[metric]
                rows.append(row)
                by_profile[profile].append(row)
    experiment2.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["lead_profile", "outer_fold", "seed"]
    with (experiment2 / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    profiles = {}
    for profile in ("L1", "L2"):
        profile_rows = by_profile[profile]
        profiles[profile] = {
            "status": "complete" if len(profile_rows) == 15 else "pending" if not profile_rows else "in_progress",
            "run_count": len(profile_rows),
            "expected_run_count": 15,
            "record_macro": {metric: aggregate([row[f"outer_record_macro_{metric}"] for row in profile_rows if row[f"outer_record_macro_{metric}"] is not None]) for metric in METRICS},
            "pooled": {metric: aggregate([row[f"outer_pooled_{metric}"] for row in profile_rows if row[f"outer_pooled_{metric}"] is not None]) for metric in METRICS},
            "parameter_count": sorted({row["parameter_count"] for row in profile_rows if row["parameter_count"] is not None}),
        }
    summary = {"experiment": "experiment_2_one_vs_two_channel_ablation", "status": "complete" if all(item["status"] == "complete" for item in profiles.values()) else "in_progress", "official_result": all(item["status"] == "complete" for item in profiles.values()), "split_sha256": SPLIT_SHA256, "selected_source_condition": "M1", "sources": ["incartdb", "mitdb"], "profiles": profiles, "external_data_accessed": False, "decision_status": "pending_until_all_L2_runs_and_target-device_resource_evidence_are_available"}
    (experiment2 / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {experiment2 / 'runs.csv'} and {experiment2 / 'summary.json'}")


if __name__ == "__main__":
    main()
