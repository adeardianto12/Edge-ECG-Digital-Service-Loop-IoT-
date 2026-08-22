"""Summarize Experiment 2.6 and apply its bounded advancement/selection rules."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np


CANDIDATES = ("P0", "P1", "P2", "P3")
FOLDS = range(1, 6)
SEEDS = (20260803, 20260804, 20260805)
SPLIT_SHA256 = "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787"


def load_run(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment") != "experiment_2_6_controlled_refinement"
        or payload.get("split_sha256") != SPLIT_SHA256
        or payload.get("sources") != ["incartdb", "mitdb"]
        or payload.get("external_data_accessed")
    ):
        raise ValueError(f"Experiment 2.6 provenance guard failed: {path}")
    return payload


def metric(payload: dict, name: str) -> float:
    macro = payload["outer_test"]["record_macro"]
    values = {
        "f1": macro["support_aware"]["pvc_f1"],
        "recall": macro["support_aware"]["pvc_recall"],
        "precision": macro["support_aware"]["pvc_precision"],
        "auprc": macro["record_macro_auprc"],
        "brier": macro["record_macro_brier_score"],
        "pvc_free_false_decisions": macro["pvc_free_records"]["false_pvc_decisions"],
    }
    return float(values[name])


def candidate_runs(output_dir: Path, candidate: str, seeds: tuple[int, ...]) -> list[dict]:
    runs = []
    for seed in seeds:
        for fold in FOLDS:
            path = output_dir / f"{candidate}_fold{fold}_seed{seed}.json"
            if path.exists():
                runs.append(load_run(path))
    return runs


def aggregate(candidate: str, runs: list[dict], expected: int) -> dict:
    row = {"candidate": candidate, "run_count": len(runs), "expected_run_count": expected}
    if not runs:
        return row
    for name in ("f1", "recall", "precision", "auprc", "brier", "pvc_free_false_decisions"):
        values = [metric(run, name) for run in runs]
        row[f"mean_{name}"] = mean(values)
        row[f"min_{name}"] = min(values)
        row[f"max_{name}"] = max(values)
    row["parameter_count_max"] = max(int(run["architecture"]["parameter_count"]) for run in runs)
    seed_means = []
    for seed in sorted({int(run["seed"]) for run in runs}):
        matching = [run for run in runs if int(run["seed"]) == seed]
        if len(matching) == 5:
            seed_means.append({"seed": seed, **{name: mean(metric(run, name) for run in matching) for name in ("f1", "recall", "precision", "auprc", "brier")}})
    row["seed_means"] = seed_means
    row["three_seed_f1_std"] = stdev([item["f1"] for item in seed_means]) if len(seed_means) == 3 else None
    row["minimum_seed_recall"] = min((item["recall"] for item in seed_means), default=None)
    return row


def record_f1(path: Path, threshold: float) -> tuple[bool, float | None, int]:
    values = np.load(path)
    labels = values["labels"].astype(int)
    decisions = values["calibrated_probability"] >= threshold
    tp = int(np.sum(decisions & (labels == 1)))
    fp = int(np.sum(decisions & (labels == 0)))
    fn = int(np.sum(~decisions & (labels == 1)))
    has_pvc = bool(np.any(labels == 1))
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return has_pvc, (2 * precision * recall / max(precision + recall, 1e-12) if has_pvc else None), fp


def paired_bootstrap(output_dir: Path, candidate: str, seeds: tuple[int, ...], resamples: int = 10000) -> dict | None:
    deltas, pvc_free_deltas = [], []
    prediction_dir = output_dir / "per_record_predictions"
    for seed in seeds:
        for fold in FOLDS:
            base_path = output_dir / f"P0_fold{fold}_seed{seed}.json"
            other_path = output_dir / f"{candidate}_fold{fold}_seed{seed}.json"
            if not base_path.exists() or not other_path.exists():
                return None
            base_threshold = float(load_run(base_path)["selection"]["threshold"])
            other_threshold = float(load_run(other_path)["selection"]["threshold"])
            prefix = f"P0_fold{fold}_seed{seed}_"
            for base_prediction in prediction_dir.glob(prefix + "*.npz"):
                suffix = base_prediction.stem[len(prefix):]
                other_prediction = prediction_dir / f"{candidate}_fold{fold}_seed{seed}_{suffix}.npz"
                if not other_prediction.exists():
                    continue
                base_has_pvc, base_f1, base_fp = record_f1(base_prediction, base_threshold)
                other_has_pvc, other_f1, other_fp = record_f1(other_prediction, other_threshold)
                if base_has_pvc and other_has_pvc:
                    deltas.append(float(other_f1 - base_f1))
                elif not base_has_pvc and not other_has_pvc:
                    pvc_free_deltas.append(other_fp - base_fp)
    if not deltas:
        return None
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(20260808)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = np.mean(values[rng.integers(0, len(values), len(values))])
    return {
        "comparison": f"{candidate}_minus_P0",
        "unit": "paired outer-test record with PVC reference",
        "resamples": resamples,
        "seed": 20260808,
        "record_count": len(values),
        "mean_f1_difference": float(np.mean(values)),
        "ci_95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "pvc_free_false_decision_difference": float(np.mean(pvc_free_deltas)) if pvc_free_deltas else None,
    }


def stage_a_decision(row: dict, control: dict) -> dict:
    if row["candidate"] == "P0":
        return {"advance": True, "reasons": ["control"]}
    if row["run_count"] != 5 or control["run_count"] != 5:
        return {"advance": False, "reasons": ["incomplete_stage_a"]}
    safety = row["mean_pvc_free_false_decisions"] <= control["mean_pvc_free_false_decisions"]
    recall = row["mean_recall"] >= 0.90
    brier = row["mean_brier"] <= control["mean_brier"] + 0.005
    noninferior_f1 = row["mean_f1"] >= control["mean_f1"] - 0.01
    parameter = row["parameter_count_max"] < 100000
    return {
        "advance": bool(safety and recall and brier and noninferior_f1 and parameter),
        "reasons": {"safety": safety, "recall": recall, "brier": brier, "f1_noninferiority": noninferior_f1, "parameter_cap": parameter},
    }


def final_decision(rows: list[dict], comparisons: dict[str, dict | None]) -> dict:
    control = next(row for row in rows if row["candidate"] == "P0")
    eligible = []
    for row in rows:
        if row["run_count"] != 15:
            continue
        comparison = comparisons.get(row["candidate"])
        safety = row["mean_pvc_free_false_decisions"] <= control["mean_pvc_free_false_decisions"]
        recall = row["mean_recall"] >= 0.90
        brier = row["mean_brier"] <= control["mean_brier"] + 0.005
        stability = row["three_seed_f1_std"] is not None and row["three_seed_f1_std"] <= 0.04
        parameter = row["parameter_count_max"] < 100000
        if row["candidate"] == "P0":
            f1_evidence = True
        elif comparison is None:
            f1_evidence = False
        else:
            f1_evidence = comparison["ci_95"][0] > 0 or (
                comparison["ci_95"][0] >= -0.01 and row["minimum_seed_recall"] > control["minimum_seed_recall"]
            )
        row["acceptance_flags"] = {"safety": safety, "mean_recall": recall, "brier": brier, "stability": stability, "parameter_cap": parameter, "paired_f1_evidence": f1_evidence}
        if all(row["acceptance_flags"].values()):
            eligible.append(row)
    if not eligible:
        return {"status": "incomplete_or_no_accepted_candidate", "selected_candidate": None}
    selected = max(eligible, key=lambda row: (
        row["minimum_seed_recall"],
        row["mean_f1"],
        row["mean_auprc"],
        -row["mean_brier"],
        -row["mean_pvc_free_false_decisions"],
        -row["parameter_count_max"],
    ))
    return {"status": "selected", "selected_candidate": selected["candidate"], "selection_order": ["minimum_seed_recall", "mean_f1", "mean_auprc", "lower_brier", "lower_pvc_free_false_burden", "lower_parameter_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiment2_6"))
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    stage_a_runs = {candidate: candidate_runs(args.output_dir, candidate, (SEEDS[0],)) for candidate in CANDIDATES}
    stage_a_rows = [aggregate(candidate, stage_a_runs[candidate], expected=5) for candidate in CANDIDATES]
    control = next(row for row in stage_a_rows if row["candidate"] == "P0")
    for row in stage_a_rows:
        row["stage_a"] = stage_a_decision(row, control)
    all_runs = {candidate: candidate_runs(args.output_dir, candidate, SEEDS) for candidate in CANDIDATES}
    rows = [aggregate(candidate, all_runs[candidate], expected=15) for candidate in CANDIDATES]
    comparisons = {candidate: paired_bootstrap(args.output_dir, candidate, SEEDS, args.bootstrap_resamples) for candidate in CANDIDATES if candidate != "P0"}
    advancing = {row["candidate"] for row in stage_a_rows if row.get("stage_a", {}).get("advance")}
    confirmed_complete = all(next(row for row in rows if row["candidate"] == candidate)["run_count"] == 15 for candidate in advancing)
    decision = final_decision(rows, comparisons) if confirmed_complete else {"status": "stage_b_incomplete", "selected_candidate": None}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flattened = []
    for row in rows:
        flattened.append({key: value for key, value in row.items() if not isinstance(value, (dict, list))})
    fields = sorted({key for row in flattened for key in row})
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
    summary = {
        "experiment": "experiment_2_6_controlled_refinement",
        "stage_a": stage_a_rows,
        "stage_b": rows,
        "paired_record_bootstrap_vs_P0": comparisons,
        "decision": decision,
        "external_data_accessed": False,
        "prohibited_sources": ["nsrdb", "svdb"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
