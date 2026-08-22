# Contributing

This repository supports a research manuscript and a reproducible software
artifact. Contributions must preserve the evidence boundaries in `AGENTS.md`.

Before submitting a change:

- Do not add ECG files, patient data, model caches, or private experiment logs.
- Keep SVDB and NSRDB out of training, calibration, threshold, and architecture-selection paths.
- Record seeds, configurations, source hashes, and the evidence class for new experiments.
- Label replay-based service-loop results as simulated; do not imply hardware or clinical evidence.
- Run `python -m unittest discover -s tests -v` and `python -m compileall src tests`.

Use focused pull requests. Changes that alter a frozen model, threshold, lead
mapping, or preprocessing contract require a new declared experiment and must
not overwrite historical artifacts.
