"""Summarize confirmed Experiment 2.5 candidates across the three fixed seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


SEEDS = (20260803, 20260804, 20260805)
FOLDS = range(1, 6)
TARGETS = {
    "support_aware_f1": (0.80, ">="),
    "support_aware_recall": (0.90, ">="),
    "support_aware_precision": (0.75, ">="),
    "record_macro_auprc": (0.88, ">="),
    "record_macro_brier_score": (0.04, "<="),
    "legacy_zero_filled_macro_pvc_f1": (0.70, ">="),
}


def value(payload: dict, name: str) -> float:
    metric = payload["outer_test"]["record_macro"]
    lookup = {
        "support_aware_f1": metric["support_aware"]["pvc_f1"],
        "support_aware_recall": metric["support_aware"]["pvc_recall"],
        "support_aware_precision": metric["support_aware"]["pvc_precision"],
        "record_macro_auprc": metric["record_macro_auprc"],
        "record_macro_brier_score": metric["record_macro_brier_score"],
        "legacy_zero_filled_macro_pvc_f1": metric["legacy_zero_filled_macro_pvc_f1"],
    }
    return float(lookup[name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="O1")
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiment2_5"))
    args = parser.parse_args()
    seed_rows = []
    for seed in SEEDS:
        runs = []
        for fold in FOLDS:
            path = args.output_dir / f"{args.candidate}_fold{fold}_seed{seed}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("sources") != ["incartdb", "mitdb"] or payload.get("external_data_accessed") or payload.get("split_sha256") != "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787":
                raise ValueError(f"Provenance failure: {path}")
            runs.append(payload)
        seed_rows.append({"seed": seed, **{name: mean(value(run, name) for run in runs) for name in TARGETS}})
    aggregate = {}
    for name in TARGETS:
        values = [row[name] for row in seed_rows]
        aggregate[name] = {"mean": mean(values), "std": stdev(values), "min": min(values), "max": max(values)}
    flags = {name: aggregate[name]["mean"] >= target if operator == ">=" else aggregate[name]["mean"] <= target for name, (target, operator) in TARGETS.items()}
    output = {
        "experiment": "experiment_2_5_pre_freeze_model_optimization", "stage": "B", "candidate": args.candidate,
        "status": "confirmed", "seeds": list(SEEDS), "folds_per_seed": 5, "seed_rows": seed_rows,
        "aggregate": aggregate, "target_flags": flags, "all_targets_met": all(flags.values()),
        "source_boundary": {"sources": ["incartdb", "mitdb"], "external_data_accessed": False, "prohibited_sources": ["nsrdb", "svdb"]},
        "scope_note": f"This confirms {args.candidate} only. Experiment 2.5 remains incomplete until the final preregistered ladder decision is resolved.",
    }
    path = args.output_dir / f"stage_b_{args.candidate}_summary.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(path), "all_targets_met": output["all_targets_met"], "target_flags": flags}, indent=2))


if __name__ == "__main__":
    main()
