"""Create a reproducible audit manifest for the multi-source PVC experiment.

The manifest records only metadata and label counts; it never exports ECG
waveforms.  It fixes the source roles, the 360 Hz preprocessing contract, and
the lead-selection rules before any expanded-model training takes place.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import wfdb

from multisource_ecg import (
    ACCEPTED_SYMBOLS,
    HALF_WINDOW_SAMPLES,
    SOURCE_SPECS,
    TARGET_SAMPLE_RATE_HZ,
    WINDOW_SIZE_SAMPLES,
    SourceSpec,
    resampling_contract,
    select_leads,
)

MITBIH_SHARED_SUBJECT_GROUP = {"201", "202"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_label_counts(annotation, source_hz: float, signal_length: int) -> tuple[Counter, Counter]:
    accepted = Counter()
    eligible = Counter()
    resampled_length = math.ceil(signal_length * TARGET_SAMPLE_RATE_HZ / source_hz)
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in ACCEPTED_SYMBOLS:
            continue
        accepted[symbol] += 1
        resampled_sample = int(round(sample * TARGET_SAMPLE_RATE_HZ / source_hz))
        if (
            resampled_sample >= HALF_WINDOW_SAMPLES
            and resampled_sample + HALF_WINDOW_SAMPLES < resampled_length
        ):
            eligible[symbol] += 1
    return accepted, eligible


def split_group(source_key: str, record_name: str) -> str:
    if source_key == "mitdb" and record_name in MITBIH_SHARED_SUBJECT_GROUP:
        return "mitdb:201_202"
    return f"{source_key}:{record_name}"


def audit_record(spec: SourceSpec, header_path: Path, include_hashes: bool) -> dict:
    record_base = header_path.with_suffix("")
    header = wfdb.rdheader(str(record_base))
    signal_names = list(header.sig_name)
    single_lead, two_lead = select_leads(signal_names, spec)

    # Read one sample from each selected channel, which verifies that the local
    # waveform data are decodable without loading the full long-term record.
    wfdb.rdrecord(str(record_base), sampto=1, channels=two_lead["channel_indices"])
    annotation = wfdb.rdann(str(record_base), "atr")
    raw_accepted, eligible = eligible_label_counts(annotation, header.fs, header.sig_len)
    symbols = Counter(annotation.symbol)

    file_hashes = {}
    if include_hashes:
        for suffix in (".hea", ".dat", ".atr"):
            path = record_base.with_suffix(suffix)
            if not path.is_file():
                raise FileNotFoundError(f"Missing required WFDB file: {path}")
            file_hashes[suffix[1:]] = sha256_file(path)

    return {
        "source": spec.key,
        "source_description": spec.description,
        "role": spec.role,
        "record": header_path.stem,
        "split_group": split_group(spec.key, header_path.stem),
        "source_sample_rate_hz": header.fs,
        "canonical_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "resampling": resampling_contract(header.fs),
        "native_signal_length_samples": header.sig_len,
        "native_duration_seconds": header.sig_len / header.fs,
        "available_leads": signal_names,
        "single_lead": single_lead,
        "two_lead": two_lead,
        "annotation_count": len(annotation.sample),
        "annotation_symbols": dict(sorted(symbols.items())),
        "accepted_labels_before_window_boundary_check": {symbol: raw_accepted[symbol] for symbol in sorted(ACCEPTED_SYMBOLS)},
        "eligible_labels": {symbol: eligible[symbol] for symbol in sorted(ACCEPTED_SYMBOLS)},
        "excluded_annotation_count": len(annotation.sample) - sum(raw_accepted.values()),
        "file_sha256": file_hashes,
    }


def source_summary(rows: list[dict], spec: SourceSpec) -> dict:
    eligible = Counter()
    accepted = Counter()
    lead_fallbacks = {"single": [], "two": []}
    for row in rows:
        eligible.update(row["eligible_labels"])
        accepted.update(row["accepted_labels_before_window_boundary_check"])
        if row["single_lead"]["fallback_used"]:
            lead_fallbacks["single"].append(row["record"])
        if row["two_lead"]["fallback_used"]:
            lead_fallbacks["two"].append(row["record"])
    return {
        "source": spec.key,
        "description": spec.description,
        "role": spec.role,
        "record_count": len(rows),
        "accepted_labels_before_window_boundary_check": {symbol: accepted[symbol] for symbol in sorted(ACCEPTED_SYMBOLS)},
        "eligible_labels": {symbol: eligible[symbol] for symbol in sorted(ACCEPTED_SYMBOLS)},
        "single_lead_fallback_records": lead_fallbacks["single"],
        "two_lead_fallback_records": lead_fallbacks["two"],
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "source", "role", "record", "split_group", "source_sample_rate_hz", "canonical_sample_rate_hz",
        "native_signal_length_samples", "native_duration_seconds", "available_leads", "single_leads",
        "single_fallback_used", "two_leads", "two_fallback_used", "annotation_count", "eligible_N",
        "eligible_V", "accepted_N_before_boundary_check", "accepted_V_before_boundary_check",
        "excluded_annotation_count", "header_sha256", "signal_sha256", "annotation_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            hashes = row["file_sha256"]
            writer.writerow({
                "source": row["source"],
                "role": row["role"],
                "record": row["record"],
                "split_group": row["split_group"],
                "source_sample_rate_hz": row["source_sample_rate_hz"],
                "canonical_sample_rate_hz": row["canonical_sample_rate_hz"],
                "native_signal_length_samples": row["native_signal_length_samples"],
                "native_duration_seconds": f"{row['native_duration_seconds']:.6f}",
                "available_leads": ";".join(row["available_leads"]),
                "single_leads": ";".join(row["single_lead"]["lead_names"]),
                "single_fallback_used": row["single_lead"]["fallback_used"],
                "two_leads": ";".join(row["two_lead"]["lead_names"]),
                "two_fallback_used": row["two_lead"]["fallback_used"],
                "annotation_count": row["annotation_count"],
                "eligible_N": row["eligible_labels"]["N"],
                "eligible_V": row["eligible_labels"]["V"],
                "accepted_N_before_boundary_check": row["accepted_labels_before_window_boundary_check"]["N"],
                "accepted_V_before_boundary_check": row["accepted_labels_before_window_boundary_check"]["V"],
                "excluded_annotation_count": row["excluded_annotation_count"],
                "header_sha256": hashes.get("hea", ""),
                "signal_sha256": hashes.get("dat", ""),
                "annotation_sha256": hashes.get("atr", ""),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local multi-source ECG datasets for the PVC experiment")
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiment0"))
    parser.add_argument("--skip-hashes", action="store_true", help="Skip SHA-256 checksums for a quick structural audit")
    args = parser.parse_args()

    all_rows = []
    summaries = []
    for spec in SOURCE_SPECS:
        data_dir = Path(spec.directory)
        headers = sorted(data_dir.glob("*.hea"))
        if not headers:
            raise FileNotFoundError(f"No WFDB headers found for {spec.key}: {data_dir}")
        rows = [audit_record(spec, header_path, not args.skip_hashes) for header_path in headers]
        all_rows.extend(rows)
        summaries.append(source_summary(rows, spec))

    all_rows.sort(key=lambda row: (row["source"], row["record"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(all_rows, args.output_dir / "record_manifest.csv")
    contract = {
        "experiment": "experiment_0_multisource_audit",
        "task": "normal N versus PVC V beat classification",
        "accepted_labels": {"0": "normal_N", "1": "PVC_V"},
        "excluded_labels": "all WFDB annotation symbols except N and V",
        "canonical_preprocessing": {
            "target_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
            "resampling_method": "scipy.signal.resample_poly",
            "window_size_samples": WINDOW_SIZE_SAMPLES,
            "half_window_samples": HALF_WINDOW_SAMPLES,
            "window_duration_seconds": WINDOW_SIZE_SAMPLES / TARGET_SAMPLE_RATE_HZ,
            "per_window_normalisation": "zero mean and unit standard deviation, independently for each input channel",
        },
        "data_role_contract": {
            "training_development": ["mitdb", "incartdb", "ltdb"],
            "locked_external_test": ["svdb"],
            "normal_rhythm_evaluation_only": ["nsrdb"],
            "prohibited": "SVDB and NSRDB records must not enter PVC classifier fitting, threshold selection, or hyperparameter selection.",
        },
        "split_contract": {
            "unit": "record-level split_group",
            "mitdb_shared_subject_group": "Records 201 and 202 share split_group mitdb:201_202 and must remain together.",
        },
        "hashes_included": not args.skip_hashes,
        "source_summaries": summaries,
        "records": all_rows,
    }
    (args.output_dir / "preprocessing_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "source_summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
