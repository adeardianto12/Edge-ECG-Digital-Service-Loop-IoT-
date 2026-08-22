"""Summarize completed Experiment 2.6R Stage A without rerunning training."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


SPLIT_SHA256 = "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787"
CANDIDATES = ("R1", "R2", "R3")
FOLDS = range(1, 6)
SEED = 20260803
P0_STAGE_A = {
    "mean_recall": 0.9226486005987544,
    "mean_f1": 0.8305879014904098,
    "mean_auprc": 0.9117400660130631,
    "mean_brier": 0.025105891658751642,
    "pvc_free_fp_total": 161,
    "parameter_count": 3506,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment") != "experiment_2_6r_recall_stability_revision"
        or payload.get("split_sha256") != SPLIT_SHA256
        or payload.get("sources") != ["incartdb", "mitdb"]
        or payload.get("external_data_accessed")
    ):
        raise ValueError(f"2.6R provenance guard failed: {path}")
    return payload


def metric(payload: dict, name: str) -> float:
    macro = payload["outer_test"]["record_macro"]
    values = {
        "recall": macro["support_aware"]["pvc_recall"],
        "f1": macro["support_aware"]["pvc_f1"],
        "auprc": macro["record_macro_auprc"],
        "brier": macro["record_macro_brier_score"],
        "pvc_free_fp": macro["pvc_free_records"]["false_pvc_decisions"],
    }
    return float(values[name])


def main() -> None:
    output_dir = Path("results/experiment2_6r")
    rows, summaries = [], []
    for candidate in CANDIDATES:
        runs = [load_run(output_dir / f"{candidate}_fold{fold}_seed{SEED}.json") for fold in FOLDS]
        if len(runs) != 5:
            raise ValueError(f"Incomplete Stage A candidate: {candidate}")
        for run in runs:
            rows.append({
                "candidate": candidate,
                "fold": run["outer_fold"],
                "seed": run["seed"],
                "support_aware_recall": metric(run, "recall"),
                "support_aware_f1": metric(run, "f1"),
                "record_macro_auprc": metric(run, "auprc"),
                "record_macro_brier": metric(run, "brier"),
                "pvc_free_false_decisions": metric(run, "pvc_free_fp"),
                "parameter_count": run["architecture"]["parameter_count"],
                "architecture_sha256": run["architecture"]["json_sha256"],
                "r3_inner_safety_feasible": run["selection"].get("r3_feasible"),
                "external_data_accessed": run["external_data_accessed"],
            })
        summary = {
            "candidate": candidate,
            "run_count": len(runs),
            "mean_recall": mean(metric(run, "recall") for run in runs),
            "mean_f1": mean(metric(run, "f1") for run in runs),
            "mean_auprc": mean(metric(run, "auprc") for run in runs),
            "mean_brier": mean(metric(run, "brier") for run in runs),
            "pvc_free_fp_total": sum(int(metric(run, "pvc_free_fp")) for run in runs),
            "parameter_count": int(runs[0]["architecture"]["parameter_count"]),
            "architecture_sha256": runs[0]["architecture"]["json_sha256"],
        }
        summary["stage_a_gates"] = {
            "safety": summary["pvc_free_fp_total"] <= P0_STAGE_A["pvc_free_fp_total"],
            "mean_recall": summary["mean_recall"] >= 0.90,
            "f1_noninferiority": summary["mean_f1"] >= P0_STAGE_A["mean_f1"] - 0.01,
            "auprc_noninferiority": summary["mean_auprc"] >= P0_STAGE_A["mean_auprc"] - 0.005,
            "brier_noninferiority": summary["mean_brier"] <= P0_STAGE_A["mean_brier"] + 0.005,
            "architecture": summary["parameter_count"] == P0_STAGE_A["parameter_count"],
        }
        summary["advance_to_stage_b"] = all(summary["stage_a_gates"].values())
        summaries.append(summary)
    fields = list(rows[0])
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "experiment": "experiment_2_6r_recall_stability_revision",
        "stage": "A_complete",
        "p0_stage_a_reference": P0_STAGE_A,
        "candidates": summaries,
        "decision": {
            "selected_candidate": "P0/O1",
            "stage_b_candidates": [],
            "reason": "No preregistered candidate passed every Stage A safety and quality gate.",
            "model_search_status": "stopped",
        },
        "external_data_accessed": False,
        "prohibited_sources": ["nsrdb", "svdb"],
        "source_code_sha256": sha256_file(Path(__file__)),
        "runner_source_code_sha256": sha256_file(Path(__file__).with_name("22_experiment2_6r_optimize.py")),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(output_dir / "summary.json"), "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
