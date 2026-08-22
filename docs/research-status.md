# Research Status

Release: `v0.1.0-research-release` (planned)

## Verified in the Local Research Record

- Experiments 0-5 have recorded artifacts.
- The software-selected configuration is M1/L2 with P0/O1.
- Gate S2 passed under the explicitly selected TensorFlow 2.15.1 runtime.
- Frozen SVDB evaluation and NSRDB false-alarm evaluation completed without
  subsequent tuning or selection.
- Automatic R-peak evaluation completed with a preserved oracle-to-automatic
  degradation analysis.

## Key Frozen Evidence

- Gate S2 support-aware record-macro PVC F1: `0.8335`.
- Gate S2 support-aware PVC recall: `0.9075`.
- Gate S2 int8 artifact size: `11,240` bytes.
- SVDB support-aware record-macro PVC F1: `0.6736`.
- NSRDB false PVC detections: `1,113` over `437.49` hours.
- NSRDB aggregate policy-v1 false service escalations: `0.606/hour`;
  record-bootstrap upper confidence limit `1.638/hour`, with worst-record
  behavior not uniformly satisfying the target.
- Automatic R-peak PVC recall: `0.7762`, compared with `0.8741` for reference
  peaks in the same frozen evaluation.

## Not Complete

Experiment 6, software-only Experiment 8A benchmark, hardware Experiment 7,
declared-hardware Experiment 8B, and Gate H remain open. The project is a
software-frozen model plus partial service-loop evidence, not a measured
physical edge healthcare deployment.
