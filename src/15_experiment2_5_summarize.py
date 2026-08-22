"""Summarize Experiment 2.5 Stage A and apply the preregistered gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


CANDIDATES = ("O0", "O1", "O2", "O3")
FOLDS = range(1, 6)
SCREENING_SEED = 20260803
TARGETS = {
    "support_aware_f1": 0.80,
    "support_aware_recall": 0.90,
    "support_aware_precision": 0.75,
    "record_macro_auprc": 0.88,
    "record_macro_brier_score": 0.04,
    "legacy_zero_filled_macro_pvc_f1": 0.70,
}


def load_runs(output_dir: Path) -> dict[str, list[dict]]:
    result = {candidate: [] for candidate in CANDIDATES}
    for candidate in CANDIDATES:
        for fold in FOLDS:
            path = output_dir / f"{candidate}_fold{fold}_seed{SCREENING_SEED}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("split_sha256") != "fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787" or payload.get("sources") != ["incartdb", "mitdb"] or payload.get("external_data_accessed"):
                    raise ValueError(f"Provenance guard failed: {path}")
                result[candidate].append(payload)
    return result


def source_metric(payload: dict, source: str, field: str) -> float | None:
    values = [metric[field] for key, metric in payload["per_record"].items() if key.startswith(source + ":") and metric.get("has_pvc_reference") and metric.get(field) is not None]
    return mean(values) if values else None


def candidate_row(candidate: str, payloads: list[dict]) -> dict:
    def values(path: tuple[str, ...]) -> list[float]:
        output = []
        for payload in payloads:
            item = payload
            for part in path:
                item = item[part]
            if item is not None:
                output.append(float(item))
        return output
    metric_paths = {
        "support_aware_f1": ("outer_test", "record_macro", "support_aware", "pvc_f1"),
        "support_aware_recall": ("outer_test", "record_macro", "support_aware", "pvc_recall"),
        "support_aware_precision": ("outer_test", "record_macro", "support_aware", "pvc_precision"),
        "record_macro_auprc": ("outer_test", "record_macro", "record_macro_auprc"),
        "record_macro_brier_score": ("outer_test", "record_macro", "record_macro_brier_score"),
        "legacy_zero_filled_macro_pvc_f1": ("outer_test", "record_macro", "legacy_zero_filled_macro_pvc_f1"),
    }
    row = {"candidate": candidate, "run_count": len(payloads), "expected_run_count": 5}
    for name, path in metric_paths.items():
        items = values(path)
        row[name] = mean(items) if items else None
        row[name + "_std"] = stdev(items) if len(items) > 1 else 0.0 if items else None
        row[name + "_min"] = min(items) if items else None
        row[name + "_max"] = max(items) if items else None
    for source in ("mitdb", "incartdb"):
        for metric in ("pvc_f1", "pvc_recall"):
            source_values = [source_metric(payload, source, metric) for payload in payloads]
            source_values = [value for value in source_values if value is not None]
            row[f"{source}_{metric}"] = mean(source_values) if source_values else None
    row["parameter_count_max"] = max((payload["architecture"]["parameter_count"] for payload in payloads), default=None)
    row["resource_screen_pass"] = bool(row["parameter_count_max"] is not None and row["parameter_count_max"] < 100_000)
    return row


def apply_gate(row: dict, baseline: dict | None) -> dict:
    target_flags = {
        "support_aware_f1": row["support_aware_f1"] is not None and row["support_aware_f1"] >= TARGETS["support_aware_f1"],
        "support_aware_recall": row["support_aware_recall"] is not None and row["support_aware_recall"] >= TARGETS["support_aware_recall"],
        "support_aware_precision": row["support_aware_precision"] is not None and row["support_aware_precision"] >= TARGETS["support_aware_precision"],
        "record_macro_auprc": row["record_macro_auprc"] is not None and row["record_macro_auprc"] >= TARGETS["record_macro_auprc"],
        "record_macro_brier_score": row["record_macro_brier_score"] is not None and row["record_macro_brier_score"] <= TARGETS["record_macro_brier_score"],
        "legacy_zero_filled_macro_pvc_f1": row["legacy_zero_filled_macro_pvc_f1"] is not None and row["legacy_zero_filled_macro_pvc_f1"] >= TARGETS["legacy_zero_filled_macro_pvc_f1"],
    }
    row["target_flags"] = target_flags
    row["all_metric_targets_met"] = bool(target_flags) and all(target_flags.values())
    if baseline is None or row["candidate"] == "O0":
        row["stage_a_advance"] = row["candidate"] == "O0"
        row["advance_reasons"] = ["baseline"] if row["candidate"] == "O0" else ["missing_baseline"]
        return row
    f1_gain = row["support_aware_f1"] - baseline["support_aware_f1"]
    recall_loss = baseline["support_aware_recall"] - row["support_aware_recall"]
    source_f1_changes = {source: row[f"{source}_pvc_f1"] - baseline[f"{source}_pvc_f1"] for source in ("mitdb", "incartdb")}
    row["support_aware_f1_gain_vs_O0"] = f1_gain
    row["support_aware_recall_loss_vs_O0"] = recall_loss
    row["source_f1_change_vs_O0"] = source_f1_changes
    criterion = f1_gain >= 0.02 and recall_loss <= 0.01 and all(change >= -0.01 for change in source_f1_changes.values()) and row["resource_screen_pass"]
    row["stage_a_advance"] = bool(row["all_metric_targets_met"] or criterion)
    row["advance_reasons"] = []
    if row["all_metric_targets_met"]:
        row["advance_reasons"].append("all_targets_met")
    if criterion:
        row["advance_reasons"].append("improvement_gate_met")
    if not row["advance_reasons"]:
        row["advance_reasons"].append("screening_gate_not_met")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiment2_5"))
    args = parser.parse_args()
    runs = load_runs(args.output_dir)
    rows = [candidate_row(candidate, runs[candidate]) for candidate in CANDIDATES]
    baseline = next((row for row in rows if row["candidate"] == "O0" and row["run_count"] == 5), None)
    for row in rows:
        if row["run_count"] == 5:
            apply_gate(row, baseline)
        else:
            row["stage_a_advance"] = False
            row["advance_reasons"] = ["incomplete_stage_a"]
    advancing = [row["candidate"] for row in rows if row.get("stage_a_advance") and row["candidate"] != "O0"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with (args.output_dir / "stage_a_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in rows])
    summary = {"experiment": "experiment_2_5_pre_freeze_model_optimization", "stage": "A", "screening_seed": SCREENING_SEED, "candidates": rows, "advancing_candidates": advancing, "o0_complete": baseline is not None, "external_data_accessed": False, "prohibited_sources": ["nsrdb", "svdb"]}
    (args.output_dir / "stage_a_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "advancing_candidates.json").write_text(json.dumps({"candidates": advancing, "external_data_accessed": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage_a_complete": baseline is not None and all(row["run_count"] == 5 for row in rows), "advancing_candidates": advancing, "summary": str(args.output_dir / "stage_a_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
