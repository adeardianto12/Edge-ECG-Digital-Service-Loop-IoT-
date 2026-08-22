# Reproducibility Guide

This release separates lightweight public code and evidence from local data
and large experiment artifacts.

## Smoke Checks Without ECG Data

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m unittest discover -s tests -v
python -m compileall src tests
python -m service_loop.run_demo
```

The demo is a contract and idempotency demonstration. It is not an Experiment
8A benchmark and does not measure physical acquisition, energy, transport, or
gateway latency.

## Frozen Software Configuration

The Gate S2 artifact was evaluated with Python 3.10 and TensorFlow 2.15.1.
Install `requirements-gate-s2.txt` for that runtime family. The frozen
development configuration is M1 (MIT-BIH + INCART), L2 (two fixed channels),
candidate P0/O1, seed `20260803`, canonical rate 360 Hz, 300-sample windows,
and threshold `0.49`. Calibration and threshold selection use development
records only.

The exact model and artifact hashes are published in
`results_public/software_freeze_manifest.json`. The full model, calibration
traces, per-record predictions, and raw data remain outside GitHub.

## Evidence Boundaries

The original Gate S pointwise-equivalence attempt failed its p99 criterion and
is retained as a negative result. Gate S2 passed its registered direct-int8
criteria under TensorFlow 2.15.1. SVDB and NSRDB were evaluated after the
software freeze, but the SVDB run is described as an isolated cross-database
holdout rather than a pristine blind test.

Experiments 7, 8A, and 8B and Gate H are not complete in this release. No
physical two-tier deployment, energy result, clinical efficacy, or regulatory
claim may be inferred from the current software-only service-loop package.
