"""Execute the one-time frozen Experiment 4 NSRDB false-alarm evaluation.

The Gate S2-passed P0/O1 int8 artifact, Platt map, and threshold are loaded
verbatim.  NSRDB is evaluation-only: this program has no training,
calibration-fitting, threshold-selection, or model-selection path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import wfdb

from multisource_ecg import SOURCE_BY_KEY
from service_loop.contracts import GatewayEvent
from service_loop.gateway_policy import PolicyV1


FROZEN_CANDIDATE = "P0/O1"
FROZEN_THRESHOLD = 0.49
EXPECTED_MODEL_SHA256 = "edea13aacdd5f6f9f94a3b73092f567f25b4dcade6133da4af7eb42aa2913776"
EXPECTED_CALIBRATION_SHA256 = "1525a1988a25021e3398a0cee5bef66263c30d31d24c159cdb730a94dcba59fa"
EXPECTED_MANIFEST_SHA256 = "6a091daa2f32fbb45b33772ef8fda7029988741cf57adb18b4e5b3f336f1c6bd"
EXPECTED_RECORD_COUNT = 18
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_RESAMPLES = 2000
ORIGIN = datetime(1970, 1, 1, tzinfo=timezone.utc)


def load_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


experiment3 = load_module("32_experiment3_frozen_svdb.py", "experiment4_experiment3")
base = experiment3.base
core = experiment3.core


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Experiment 4 artifact: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_nsrdb_records(manifest_path: Path) -> list[dict]:
    records: list[dict] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source"] != "nsrdb":
                continue
            if row["role"] != "normal_rhythm_evaluation_only":
                raise ValueError("NSRDB record does not have normal_rhythm_evaluation_only role")
            records.append(row)
    if len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError(f"Expected {EXPECTED_RECORD_COUNT} NSRDB records, found {len(records)}")
    names = [row["record"] for row in records]
    if len(names) != len(set(names)):
        raise ValueError("NSRDB manifest contains duplicate record identifiers")
    return sorted(records, key=lambda row: row["record"])


def policy_actions(record: str, canonical_samples: np.ndarray, probabilities: np.ndarray, duration_seconds: float, model_hash: str) -> dict:
    """Apply the existing policy-v1 to valid-quality, annotated-N events only."""
    policy = PolicyV1(minimum_signal_quality=0.80)
    action_counts = {"monitor": 0, "buffer_segment": 0, "request_review": 0, "reacquire": 0}
    for index, (sample, probability) in enumerate(zip(canonical_samples, probabilities)):
        event = GatewayEvent(
            event_id=f"experiment4:{record}:{index}",
            endpoint_id=f"nsrdb:{record}",
            gateway_id="experiment4-evaluation-gateway",
            sample_sequence=int(sample),
            recorded_at=(ORIGIN + timedelta(seconds=float(sample) / 360.0)).isoformat(),
            lead_profile="two_channel",
            native_sample_rate_hz=128.0,
            transport_profile="offline_evaluation",
            model_sha256=model_hash,
            policy_version=policy.version,
            signal_quality=1.0,
            pvc_probability=float(probability),
            decision_threshold=FROZEN_THRESHOLD,
            r_peak_source="oracle_annotation",
            simulated_acquisition_endpoint=True,
        )
        action = policy.decide(event).requested_action
        action_counts[action] += 1
    hours = duration_seconds / 3600.0
    return {
        **action_counts,
        "false_service_escalations_per_hour": action_counts["request_review"] / max(hours, 1e-12),
    }


def bootstrap_record_rates(rows: list[dict]) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = ("false_positive_rate", "false_pvc_detections_per_hour", "false_service_escalations_per_hour")
    values = {name: [] for name in names}
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = [rows[int(index)] for index in rng.integers(0, len(rows), len(rows))]
        total_n = sum(row["annotated_N_beats"] for row in sampled)
        total_fp = sum(row["false_pvc_detections"] for row in sampled)
        total_hours = sum(row["duration_hours"] for row in sampled)
        total_escalations = sum(row["request_review"] for row in sampled)
        values["false_positive_rate"].append(total_fp / max(total_n, 1))
        values["false_pvc_detections_per_hour"].append(total_fp / max(total_hours, 1e-12))
        values["false_service_escalations_per_hour"].append(total_escalations / max(total_hours, 1e-12))
    return {
        name: {"bootstrap_resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "ci_95": [float(np.quantile(items, 0.025)), float(np.quantile(items, 0.975))]}
        for name, items in values.items()
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Experiment 4 artifact: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dependency_versions() -> dict:
    import sklearn
    import tensorflow as tf

    return {"python": sys.version, "platform": platform.platform(), "tensorflow": tf.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "wfdb": wfdb.__version__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/gate_s2_int8_tf215_run1/model_int8.tflite"))
    parser.add_argument("--calibration", type=Path, default=Path("models/gate_s2_int8_tf215_run1/calibration.json"))
    parser.add_argument("--freeze-manifest", type=Path, default=Path("results/gate_s2_int8_tf215_run1/manifest.json"))
    parser.add_argument("--record-manifest", type=Path, default=Path("results/experiment0/record_manifest.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--result-dir", type=Path, default=Path("results/experiment4"))
    parser.add_argument("--allow-frozen-nsrdb-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.allow_frozen_nsrdb_evaluation:
        parser.error("Experiment 4 requires --allow-frozen-nsrdb-evaluation")
    if args.result_dir.exists():
        raise FileExistsError(f"Experiment 4 output already exists: {args.result_dir}")

    model_hash = sha256_file(args.model)
    calibration_hash = sha256_file(args.calibration)
    manifest_hash = sha256_file(args.freeze_manifest)
    if model_hash != EXPECTED_MODEL_SHA256 or calibration_hash != EXPECTED_CALIBRATION_SHA256 or manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen Gate S2 artifact hash mismatch")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    freeze_manifest = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if freeze_manifest.get("status") != "passed" or freeze_manifest.get("candidate") != FROZEN_CANDIDATE:
        raise ValueError("Gate S2 manifest is not the frozen P0/O1 pass")
    if float(calibration.get("threshold", -1.0)) != FROZEN_THRESHOLD:
        raise ValueError("Frozen threshold mismatch")
    scaler = experiment3.frozen_scaler(calibration)

    args.result_dir.mkdir(parents=True)
    protocol = {
        "experiment": "experiment_4_nsrdb_normal_rhythm_false_alarm_evaluation",
        "status": "registered_execution",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": FROZEN_CANDIDATE,
        "artifact": {"path": str(args.model), "sha256": model_hash, "size_bytes": args.model.stat().st_size},
        "calibration": {"path": str(args.calibration), "sha256": calibration_hash, "method": calibration["method"], "coefficient": calibration["coefficient"], "intercept": calibration["intercept"], "threshold": FROZEN_THRESHOLD},
        "freeze_manifest": {"path": str(args.freeze_manifest), "sha256": manifest_hash, "status": freeze_manifest["status"]},
        "external_source": "nsrdb",
        "external_role": "normal_rhythm_evaluation_only",
        "input_scope": "annotated N beats only for false-alarm metrics; the 26 annotated V beats are excluded from normal-beat denominators and policy-event counts",
        "preprocessing": "fixed 360 Hz polyphase resampling, 300-sample centered oracle-annotation windows, per-window/channel normalization, frozen past-only RR features",
        "policy": {"version": "policy-v1", "minimum_signal_quality": 0.80, "evaluation_signal_quality": 1.0, "aggregation": "existing PolicyV1: each threshold event is buffer_segment until the rolling 30-second high-risk deque reaches three events, after which each in-window threshold event requests review"},
        "unavailable_measurements": {"signal_quality_relationship": "not measured: Experiment 4 uses valid-quality oracle windows", "r_peak_error_relationship": "not measured: automatic R-peak evaluation is reserved for Experiment 5"},
        "prohibited_operations": ["retraining", "calibration fitting", "threshold selection", "model selection", "quantization change", "lead-profile change", "preprocessing change"],
        "bootstrap": {"unit": "record", "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "write_once": True,
    }
    write_json(args.result_dir / "protocol.json", protocol)

    rows, prediction_arrays = [], {}
    for index, record in enumerate(load_nsrdb_records(args.record_manifest), start=1):
        record_base = args.data_root / SOURCE_BY_KEY["nsrdb"].directory / record["record"]
        accepted, rr = experiment3.causal_rr_features(record_base, lead_count=2)
        raw = base.tflite_predict(args.model, accepted["waveforms"], rr).astype(np.float32)
        probability = core.apply_scaler(scaler, raw).astype(np.float32)
        normal = accepted["labels"] == 0
        header = wfdb.rdheader(str(record_base))
        duration_seconds = float(header.sig_len / header.fs)
        normal_probability = probability[normal]
        normal_raw = raw[normal]
        normal_samples = accepted["canonical_r_peak_samples"][normal]
        actions = policy_actions(record["record"], normal_samples, normal_probability, duration_seconds, model_hash)
        false_pvc = int(np.sum(normal_probability >= FROZEN_THRESHOLD))
        hours = duration_seconds / 3600.0
        row = {
            "record_key": f"nsrdb:{record['record']}", "duration_seconds": duration_seconds, "duration_hours": hours,
            "annotated_N_beats": int(np.sum(normal)), "annotated_V_beats_excluded": int(np.sum(~normal)),
            "false_pvc_detections": false_pvc, "false_positive_rate": false_pvc / max(int(np.sum(normal)), 1),
            "false_pvc_detections_per_hour": false_pvc / max(hours, 1e-12), **actions,
        }
        rows.append(row)
        prefix = f"nsrdb__{record['record']}"
        prediction_arrays[f"{prefix}__canonical_r_peak_samples_N"] = normal_samples.astype(np.int64)
        prediction_arrays[f"{prefix}__raw_int8_probability_N"] = normal_raw
        prediction_arrays[f"{prefix}__calibrated_probability_N"] = normal_probability
        print(f"[{index}/{EXPECTED_RECORD_COUNT}] nsrdb:{record['record']}: {row['annotated_N_beats']} annotated N beats", flush=True)

    total_n = sum(row["annotated_N_beats"] for row in rows)
    total_fp = sum(row["false_pvc_detections"] for row in rows)
    total_hours = sum(row["duration_hours"] for row in rows)
    total_escalations = sum(row["request_review"] for row in rows)
    worst_record = max(rows, key=lambda row: (row["false_service_escalations_per_hour"], row["false_pvc_detections_per_hour"], row["record_key"]))
    aggregate = {
        "annotated_N_beats": total_n,
        "annotated_V_beats_excluded": sum(row["annotated_V_beats_excluded"] for row in rows),
        "total_duration_hours": total_hours,
        "false_pvc_detections": total_fp,
        "false_positive_rate": total_fp / max(total_n, 1),
        "false_pvc_detections_per_hour": total_fp / max(total_hours, 1e-12),
        "false_service_escalations": total_escalations,
        "false_service_escalations_per_hour": total_escalations / max(total_hours, 1e-12),
        "engineering_target_service_escalations_per_hour_lte": 1.0,
        "engineering_target_pass": (total_escalations / max(total_hours, 1e-12)) <= 1.0,
    }
    write_csv(args.result_dir / "per_record_false_alarms.csv", rows)
    np.savez_compressed(args.result_dir / "per_record_predictions_N.npz", **prediction_arrays)
    result = {
        "experiment": "experiment_4_nsrdb_normal_rhythm_false_alarm_evaluation",
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_data_accessed": True, "external_source": "nsrdb", "frozen_configuration_verified": True,
        "candidate": FROZEN_CANDIDATE, "artifact_sha256": model_hash, "calibration_sha256": calibration_hash,
        "freeze_manifest_sha256": manifest_hash, "threshold": FROZEN_THRESHOLD, "record_count": len(rows),
        "aggregate_false_alarm_metrics": aggregate, "record_level_bootstrap_ci_95": bootstrap_record_rates(rows),
        "worst_record": worst_record,
        "relationship_to_signal_quality_and_r_peak_errors": protocol["unavailable_measurements"],
        "artifacts": {"protocol": str(args.result_dir / "protocol.json"), "per_record_false_alarms": str(args.result_dir / "per_record_false_alarms.csv"), "per_record_predictions_N": str(args.result_dir / "per_record_predictions_N.npz")},
        "source_sha256": sha256_file(Path(__file__)), "dependencies": dependency_versions(),
        "post_evaluation_rule": "No model, threshold, calibration, quantization, preprocessing, lead-profile, or seed change may be selected from these NSRDB results.",
    }
    write_json(args.result_dir / "summary.json", result)
    print(json.dumps({"status": result["status"], "summary": str(args.result_dir / "summary.json"), "record_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
