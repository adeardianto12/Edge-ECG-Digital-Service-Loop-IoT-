# Two-Tier Architecture

```text
ECG acquisition endpoint
  sensing, timestamps, sequence numbers, bounded buffer
        |
        | frozen local transport profile
        v
Local edge gateway
  stream reconstruction, quality, resampling, causal R-peaks,
  int8 inference, policy, audit, acknowledgement
        |
        v
Endpoint acknowledgement and reacquisition command
```

The latency-critical loop is local and must not require Internet or cloud
inference. A replay endpoint and in-memory transport are deliberately labelled
simulation components. They provide contract and fault-injection scaffolding,
not physical acquisition or hardware performance evidence.

The gateway policy is non-diagnostic. It can continue monitoring, request
reacquisition, buffer a segment, or request human review after the frozen
multi-event policy is satisfied.
