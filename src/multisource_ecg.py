"""Shared multi-source ECG preprocessing for the PVC experiments.

This module implements the Experiment 0 contract.  It deliberately excludes
split construction, model fitting, calibration, and any use of SVDB/NSRDB in
training decisions; those are responsibilities of later experiment scripts.
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE_HZ = 360
WINDOW_SIZE_SAMPLES = 300
HALF_WINDOW_SAMPLES = WINDOW_SIZE_SAMPLES // 2
ACCEPTED_SYMBOLS = {"N", "V"}


@dataclass(frozen=True)
class SourceSpec:
    key: str
    directory: str
    role: str
    single_preferred: tuple[str, ...]
    two_preferred: tuple[str, ...]
    description: str


SOURCE_SPECS = (
    SourceSpec("mitdb", "data/mitdb", "training_development", ("MLII",), ("MLII",), "MIT-BIH Arrhythmia Database"),
    SourceSpec("incartdb", "data/incartdb", "training_development", ("II",), ("II", "V1"), "St Petersburg INCART 12-lead Arrhythmia Database"),
    SourceSpec("ltdb", "data/ltdb", "training_development", ("ECG1",), ("ECG1", "ECG2"), "MIT-BIH Long-Term ECG Database"),
    SourceSpec("svdb", "data/svdb", "locked_external_test", ("ECG1",), ("ECG1", "ECG2"), "MIT-BIH Supraventricular Arrhythmia Database"),
    SourceSpec("nsrdb", "data/nsrdb", "normal_rhythm_evaluation_only", ("ECG1",), ("ECG1", "ECG2"), "MIT-BIH Normal Sinus Rhythm Database"),
)
SOURCE_BY_KEY = {spec.key: spec for spec in SOURCE_SPECS}


def select_leads(signal_names: list[str], spec: SourceSpec) -> tuple[dict, dict]:
    """Return auditable one- and two-channel mappings for one record."""
    if not signal_names:
        raise ValueError("Record has no signal channels")

    single_name = next((name for name in spec.single_preferred if name in signal_names), signal_names[0])
    single = {
        "lead_names": [single_name],
        "channel_indices": [signal_names.index(single_name)],
        "fallback_used": single_name not in spec.single_preferred,
    }

    selected = [name for name in spec.two_preferred if name in signal_names]
    for name in signal_names:
        if len(selected) == 2:
            break
        if name not in selected:
            selected.append(name)
    if len(selected) < 2:
        raise ValueError("Record does not provide two channels")
    two = {
        "lead_names": selected[:2],
        "channel_indices": [signal_names.index(name) for name in selected[:2]],
        "fallback_used": any(name not in selected[:2] for name in spec.two_preferred),
    }
    return single, two


def resampling_contract(source_hz: float) -> dict:
    ratio = Fraction(TARGET_SAMPLE_RATE_HZ / source_hz).limit_denominator(10000)
    return {
        "method": "scipy.signal.resample_poly",
        "source_hz": source_hz,
        "target_hz": TARGET_SAMPLE_RATE_HZ,
        "up": ratio.numerator,
        "down": ratio.denominator,
    }


def resample_channels(signal: np.ndarray, source_hz: float) -> np.ndarray:
    """Resample time-by-channel ECG data to the canonical rate."""
    if source_hz == TARGET_SAMPLE_RATE_HZ:
        return signal.astype(np.float32, copy=False)
    ratio = Fraction(TARGET_SAMPLE_RATE_HZ / source_hz).limit_denominator(10000)
    return resample_poly(signal, ratio.numerator, ratio.denominator, axis=0).astype(np.float32)


def normalise_window(window: np.ndarray) -> np.ndarray:
    """Apply independent zero-mean, unit-variance scaling to each channel."""
    mean = window.mean(axis=0, keepdims=True)
    standard_deviation = window.std(axis=0, keepdims=True)
    return ((window - mean) / np.maximum(standard_deviation, 1e-8)).astype(np.float32)


def load_pvc_windows(record_base: Path, spec: SourceSpec, lead_count: int = 1) -> dict:
    """Load one record as canonical PVC windows under the Experiment 0 contract.

    The returned labels use 0 for ``N`` and 1 for ``V``.  Other annotations
    are excluded and no data are silently padded at record boundaries.
    """
    if lead_count not in (1, 2):
        raise ValueError("lead_count must be 1 or 2")
    header = wfdb.rdheader(str(record_base))
    single, two = select_leads(list(header.sig_name), spec)
    lead_mapping = single if lead_count == 1 else two
    record = wfdb.rdrecord(str(record_base), channels=lead_mapping["channel_indices"])
    annotation = wfdb.rdann(str(record_base), "atr")
    signal = record.p_signal
    signal = resample_channels(signal, record.fs)

    windows, labels, canonical_peaks, original_peaks = [], [], [], []
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in ACCEPTED_SYMBOLS:
            continue
        canonical_sample = int(round(sample * TARGET_SAMPLE_RATE_HZ / record.fs))
        if (
            canonical_sample < HALF_WINDOW_SAMPLES
            or canonical_sample + HALF_WINDOW_SAMPLES >= len(signal)
        ):
            continue
        window = signal[
            canonical_sample - HALF_WINDOW_SAMPLES : canonical_sample + HALF_WINDOW_SAMPLES
        ]
        windows.append(normalise_window(window))
        labels.append(0 if symbol == "N" else 1)
        canonical_peaks.append(canonical_sample)
        original_peaks.append(int(sample))

    return {
        "waveforms": np.asarray(windows, dtype=np.float32).reshape((-1, WINDOW_SIZE_SAMPLES, lead_count)),
        "labels": np.asarray(labels, dtype=np.int32),
        "canonical_r_peak_samples": np.asarray(canonical_peaks, dtype=np.int64),
        "original_annotation_samples": np.asarray(original_peaks, dtype=np.int64),
        "record": record_base.name,
        "source": spec.key,
        "source_sample_rate_hz": record.fs,
        "selected_leads": lead_mapping["lead_names"],
        "lead_fallback_used": lead_mapping["fallback_used"],
    }
