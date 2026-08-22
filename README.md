# Edge ECG Digital Service Loop

Research code and lightweight evidence for an auditable PVC-risk service loop
between an ECG acquisition endpoint and a local edge gateway. The intended
submission target is IEEE Internet of Things Magazine.

> This is research software, not a medical device or diagnostic system. It
> does not provide clinical efficacy, regulatory, or therapeutic claims.

```text
ECG acquisition endpoint
  sensing, timestamps, sequence numbers, bounded buffer
        |
        v
Local edge gateway
  signal quality, causal R peaks, int8 PVC inference, local policy, audit
        |
        v
Endpoint acknowledgement: monitor, reacquire, buffer, or request review
```

The latency-critical loop is designed to be local and cloud-independent. The
current endpoint replay and in-memory transport components are simulations;
they do not establish physical acquisition, energy, or hardware latency.

## Research Status

The published software evidence includes Experiment 0-5 records, the passed
Gate S2 direct-int8 software freeze, frozen SVDB evaluation, NSRDB false-alarm
evaluation, and the automatic R-peak analysis. The selected frozen software
configuration is P0/O1, M1 (MIT-BIH + INCART), and L2 (two fixed channels),
with a threshold of `0.49`.

The original Gate S pointwise p99 criterion failed and remains a disclosed
negative result. Gate S2 passed under TensorFlow 2.15.1. SVDB is reported as
an isolated cross-database holdout, not a pristine blind test.

Experiment 6, Experiment 8A, Experiment 7, Experiment 8B, and Gate H are not
complete. This release must not be described as a measured physical two-tier
edge healthcare deployment.

| Evidence | Current result |
|---|---|
| Gate S2 development F1 | Support-aware record-macro PVC F1 `0.8335` |
| Gate S2 development recall | Support-aware record-macro PVC recall `0.9075` |
| Frozen int8 model | `11,240` bytes; held float/int8 decision agreement `1.000` |
| SVDB external evaluation | Support-aware record-macro PVC F1 `0.6736` |
| NSRDB policy-v1 false escalation | `0.606/hour` overall; upper bootstrap CI `1.638/hour` |
| Automatic R-peak evaluation | PVC recall `0.7762` versus `0.8741` with reference peaks |

Read [research status](docs/research-status.md), the
[experiment matrix](docs/experiment-matrix.md), and the full research charter
in [AGENTS.md](AGENTS.md) before interpreting these values.

## Quick Start

Use Python 3.10. The following checks require no ECG download:

```powershell
python -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "$PWD/src"
python -m unittest discover -s tests -v
python -m compileall src tests
python -m service_loop.run_demo
```

The demo verifies schema handling, idempotency, sequence checks, model-hash
rejection, and the policy-v1 persistent-risk action. It is a non-benchmark
contract demonstration, not an Experiment 8A result.

For the passed Gate S2 runtime family, install
`requirements-gate-s2.txt`. It pins TensorFlow `2.15.1`; the general
development requirements intentionally do not claim to reproduce Gate S2.

## Data and Artifacts

Raw ECG data are not distributed. Obtain MIT-BIH, INCART, LTDDB, SVDB, and
NSRDB from their official sources and follow their terms and citations. See
[DATASETS.md](DATASETS.md).

GitHub contains code, protocols, documentation, and small reviewed evidence
summaries under [results_public](results_public). Full results, window caches,
per-record predictions, raw data, training logs, and unapproved weights remain
in a restricted research archive. A public GitHub Release is not private
storage.

## Reproduction Boundaries

The Gate S2 release freezes:

- Python 3.10, TensorFlow 2.15.1, seed `20260803`.
- M1 development sources: MIT-BIH plus INCART.
- L2 two-channel input, 360 Hz canonical timing, and 300-sample R-centered windows.
- P0/O1, source- and record-balanced Platt calibration, and threshold `0.49`.
- No SVDB or NSRDB data in training, calibration, threshold, lead, seed, or model selection.

The small public manifest supplies model and calibration hashes without
distributing restricted artifacts. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
and [results_public/software_freeze_manifest.json](results_public/software_freeze_manifest.json).

## Repository Layout

`src/` contains training, evaluation, and service-loop code; `tests/` contains
data-independent contract tests; `results_public/` contains reviewed summaries;
`docs/` contains release documentation; and `AGENTS.md` is the research charter.

## Contributing and Citation

Contribution rules are in [CONTRIBUTING.md](CONTRIBUTING.md). Cite the
associated manuscript and this repository using [CITATION.cff](CITATION.cff).

Source code is licensed under [Apache-2.0](LICENSE). Project documentation,
protocols, and original figures are intended for release under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), except where a file
states otherwise. Third-party datasets retain their own terms.
