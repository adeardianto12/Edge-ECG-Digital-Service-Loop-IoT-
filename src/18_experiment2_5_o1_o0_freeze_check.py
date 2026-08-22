"""Complete the preregistered O1-versus-O0 development freeze checks."""

import json
from pathlib import Path

import numpy as np


SEED = 20260803
FOLDS = range(1, 6)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260807


def threshold(output_dir: Path, candidate: str, fold: int) -> float:
    payload = json.loads((output_dir / f"{candidate}_fold{fold}_seed{SEED}.json").read_text(encoding="utf-8"))
    if payload["sources"] != ["incartdb", "mitdb"] or payload["external_data_accessed"]:
        raise ValueError(f"Provenance failure for {candidate} fold {fold}")
    return float(payload["selection"]["threshold"])


def metrics(path: Path, decision_threshold: float) -> dict:
    payload = np.load(path)
    labels = payload["labels"].astype(int)
    decisions = payload["calibrated_probability"] >= decision_threshold
    true_positive = int(np.sum(decisions & (labels == 1)))
    false_positive = int(np.sum(decisions & (labels == 0)))
    true_negative = int(np.sum(~decisions & (labels == 0)))
    false_negative = int(np.sum(~decisions & (labels == 1)))
    has_pvc = bool(np.any(labels == 1))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "has_pvc": has_pvc,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12) if has_pvc else None,
        "false_positive": false_positive,
        "specificity": true_negative / max(true_negative + false_positive, 1),
    }


def records(output_dir: Path, candidate: str) -> dict[str, dict]:
    prediction_dir = output_dir / "per_record_predictions"
    output = {}
    for fold in FOLDS:
        prefix = f"{candidate}_fold{fold}_seed{SEED}_"
        decision_threshold = threshold(output_dir, candidate, fold)
        for path in prediction_dir.glob(f"{prefix}*.npz"):
            output[path.stem[len(prefix):]] = metrics(path, decision_threshold)
    return output


def bootstrap(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_RESAMPLES)
    for index in range(BOOTSTRAP_RESAMPLES):
        means[index] = np.mean(values[rng.integers(0, len(values), len(values))])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    output_dir = Path("results/experiment2_5")
    o0, o1 = records(output_dir, "O0"), records(output_dir, "O1")
    keys = sorted(set(o0) & set(o1))
    pvc_keys = [key for key in keys if o0[key]["has_pvc"] and o1[key]["has_pvc"]]
    pvc_free_keys = [key for key in keys if not o0[key]["has_pvc"] and not o1[key]["has_pvc"]]
    differences = np.asarray([o1[key]["f1"] - o0[key]["f1"] for key in pvc_keys])
    o0_false_positive = [o0[key]["false_positive"] for key in pvc_free_keys]
    o1_false_positive = [o1[key]["false_positive"] for key in pvc_free_keys]
    output = {
        "experiment": "experiment_2_5_pre_freeze_model_optimization",
        "comparison": "O1_minus_O0",
        "seed": SEED,
        "unit": "paired outer-test record",
        "paired_record_count": len(keys),
        "pvc_record_count": len(pvc_keys),
        "pvc_free_record_count": len(pvc_free_keys),
        "mean_f1_difference_o1_minus_o0": float(np.mean(differences)),
        "paired_record_bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "ci_95": bootstrap(differences)},
        "o0_pvc_free_false_positive_mean": float(np.mean(o0_false_positive)),
        "o1_pvc_free_false_positive_mean": float(np.mean(o1_false_positive)),
        "o0_pvc_free_specificity_mean": float(np.mean([o0[key]["specificity"] for key in pvc_free_keys])),
        "o1_pvc_free_specificity_mean": float(np.mean([o1[key]["specificity"] for key in pvc_free_keys])),
        "source_boundary": {"sources": ["incartdb", "mitdb"], "external_data_accessed": False, "prohibited_sources": ["nsrdb", "svdb"]},
    }
    output["f1_ci_lower_above_zero"] = output["paired_record_bootstrap"]["ci_95"][0] > 0
    output["pvc_free_false_positive_not_increased"] = output["o1_pvc_free_false_positive_mean"] <= output["o0_pvc_free_false_positive_mean"]
    path = output_dir / "o1_vs_o0_freeze_check.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(path), "f1_ci_lower_above_zero": output["f1_ci_lower_above_zero"], "pvc_free_false_positive_not_increased": output["pvc_free_false_positive_not_increased"]}, indent=2))


if __name__ == "__main__":
    main()
