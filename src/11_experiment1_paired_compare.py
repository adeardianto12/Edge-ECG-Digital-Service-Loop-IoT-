"""Compare Experiment 1 conditions on matched held-out records only.

No waveform or database file is opened.  The analysis uses existing outer-fold
per-record metrics with identical fold and seed identifiers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


RESULTS = Path("results/experiment1")
SEEDS = (20260803, 20260804, 20260805)
FOLDS = range(1, 6)
COMPARISONS = (("M0", "M1"), ("M1", "M2"))
METRICS = ("pvc_f1", "pvc_recall", "auprc", "brier_score")
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_RESAMPLES = 1000


def load_result(condition: str, fold: int, seed: int) -> dict:
    path = RESULTS / f"{condition}_fold{fold}_seed{seed}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("official_result") or result.get("status") != "complete":
        raise ValueError(f"Result is not a complete official run: {path}")
    return result


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        means[index] = values[rng.integers(0, len(values), size=len(values))].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    summaries = {}
    for reference, candidate in COMPARISONS:
        comparison_name = f"{candidate}_minus_{reference}"
        matched: dict[str, list[float]] = {metric: [] for metric in METRICS}
        run_rows = []
        for fold in FOLDS:
            for seed in SEEDS:
                baseline = load_result(reference, fold, seed)["per_record"]
                challenger = load_result(candidate, fold, seed)["per_record"]
                keys = sorted(set(baseline) & set(challenger))
                if not keys:
                    raise ValueError(f"No matched records for {comparison_name}, fold {fold}, seed {seed}")
                for metric in METRICS:
                    values = np.asarray(
                        [
                            challenger[key][metric] - baseline[key][metric]
                            for key in keys
                            if challenger[key][metric] is not None and baseline[key][metric] is not None
                        ],
                        dtype=np.float64,
                    )
                    if not len(values):
                        continue
                    matched[metric].extend(values.tolist())
                    run_rows.append({
                        "comparison": comparison_name,
                        "fold": fold,
                        "seed": seed,
                        "metric": metric,
                        "matched_record_count": len(keys),
                        "mean_difference": float(values.mean()),
                    })
        rows.extend(run_rows)
        summaries[comparison_name] = {
            "comparison_unit": "matched record-level outer-test metric",
            "candidate": candidate,
            "reference": reference,
            "run_count": len(FOLDS) * len(SEEDS),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "metrics": {
                metric: {
                    "matched_observations": len(values),
                    "mean_difference": float(np.mean(values)),
                    "bootstrap_95_ci": bootstrap_interval(np.asarray(values), rng),
                }
                for metric, values in matched.items()
            },
        }

    with (RESULTS / "paired_comparison_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "experiment": "experiment_1_multisource_training_ablation",
        "status": "complete",
        "external_data_accessed": False,
        "comparisons": summaries,
        "interpretation": "Positive values favor the candidate. This is development-only paired evidence and must not use SVDB or NSRDB.",
    }
    (RESULTS / "paired_comparison_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Wrote paired comparison CSV and summary")


if __name__ == "__main__":
    main()
