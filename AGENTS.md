# Edge ECG Digital Service Loop Research Plan

## Material Passport

- Origin Skill: `ars-codex:academic-research-suite`
- Origin Route: `academic-pipeline` Stage 1 / methodology planning
- Origin Date: 2026-08-02
- Version: `research_plan_v4`
- Last Updated: 2026-08-21
- Verification Status: `PARTIALLY VERIFIED`
- Verified Evidence: official special-issue CFP, advisor-approved two-tier edge architecture decision, local source code, Experiment 0 audit, Experiment 1/2 development reports, Experiment 2.5/2.6/2.6R and final revalidation evidence, Gate S2 direct-int8 software freeze, frozen SVDB and NSRDB reports, final automatic R-peak evaluation, and focused literature verification supporting Experiment 2.5
- Not Yet Verified: literature-level novelty, Experiment 6 robustness/fault-injection performance, declared acquisition-endpoint/local-link/gateway measurements, software-only Experiment 8A benchmark evidence, and declared-hardware Experiment 8B/Gate H local service-loop results

## 1. Purpose of This File

This file is the executable research charter for the project. It defines the research question, scope, data roles, experimental sequence, service-loop prototype, evidence gates, paper narrative, and submission schedule.

Future work must update the status table and evidence paths in this file after each experiment. Planned contributions must not be rewritten as findings until the corresponding result artifact exists and passes its gate.

## 2. Target Venue and Non-Negotiable Scope

- Venue: IEEE Internet of Things Magazine
- Special Issue: **Digital Service Loops for IoT-Enabled Connected Healthcare Systems**
- Official CFP: <https://www.comsoc.org/publications/magazines/ieee-internet-things-magazine/cfp/digital-service-loops-iot-enabled>
- Manuscript submission deadline: **31 October 2026**
- Initial decision date: 31 January 2027
- Planned publication: Third Quarter 2027

The CFP defines a digital service loop as a continuous cycle of sensing, interpretation, decision-making, execution, and validation. It lists edge/fog/cloud integration among relevant deployment patterns, but it does not require every paper to use every tier. This project adopts an advisor-approved two-tier local-edge architecture: an ECG acquisition endpoint plus a local edge gateway. Cloud computation is not required for the operational loop.

Therefore, this project must not be framed as merely a higher-accuracy ECG classifier. The PVC model is the interpretation component inside a verifiable service loop:

```text
ECG acquisition endpoint: sensing, timestamps, sequence numbers, and short local buffering
  -> local endpoint-to-gateway transport
  -> local edge gateway: signal quality, resampling, and causal R-peak interpretation
  -> calibrated PVC-risk event
  -> gateway-local policy decision
  -> safe service action
  -> acknowledgement to the acquisition endpoint and outcome validation
  -> auditable feedback and rule update
```

### Advisor-Approved Deployment Boundary

- Tier 1 is the **ECG acquisition endpoint**. It acquires one or two configured channels, attaches timing and sequence metadata, maintains a bounded short-term buffer, transmits locally, and receives monitor/reacquire acknowledgements.
- Tier 2 is the **local edge gateway**. It reconstructs the stream, evaluates signal quality, resamples to 360 Hz, detects R peaks, extracts windows, runs calibrated PVC inference, applies the versioned service policy, records the audit trail, and returns acknowledgement.
- The latency-critical sensing-to-decision-to-feedback loop must close locally between these two tiers even when Internet connectivity is absent.
- A remote caregiver or telehealth review request may be an optional outbound action after the local decision. Remote review is not a model-inference tier and is not required to close the measured local loop.
- A replay-based acquisition endpoint is acceptable for repeatable early fault injection, but it must be labelled simulated. It cannot support claims about physical electrode acquisition latency, analogue-front-end energy, or wearable deployment.

### In-Scope Service Actions

- Continue local monitoring.
- Request ECG reacquisition when signal quality is inadequate.
- Buffer and transmit a short ECG segment for remote review.
- Create a non-diagnostic caregiver or telehealth review request after a predefined event pattern.
- Record acknowledgement, delivery failure, retry, review status, and policy version.
- Return monitor/reacquire commands from the local gateway to the acquisition endpoint.

### Explicitly Out of Scope

- Autonomous medical diagnosis.
- Automatic medication, defibrillation, or other therapeutic actuation.
- Claims of clinical efficacy, regulatory approval, or patient outcome improvement.
- A model that dynamically accepts arbitrary lead counts or arbitrary lead semantics.
- Online self-training from unverified service feedback.
- Using SVDB or NSRDB to tune model architecture, calibration, thresholds, or preprocessing.
- Cloud inference, cloud orchestration, or Internet availability on the latency-critical path.
- Treating a dataset replay process as evidence of physical ECG acquisition performance.

### Software-First Sequencing and Dual-Freeze Policy

This project uses two separate evidence gates. **Gate S: Software Evaluation Freeze** fixes the architecture, development sources, seed policy, weights, calibration, threshold, lead mapping, preprocessing, int8 artifact, and manifest before frozen external model evaluation. Gate S does not require declared endpoint, transport, or gateway hardware and must never be described as an edge-deployment result.

**Gate H: Deployment Freeze** fixes the acquisition endpoint, local transport profile, gateway hardware/OS/runtime, queue configuration, and clock-synchronization method after declared-hardware measurements. Gate H requires Experiment 7 and Experiment 8B evidence before the paper may claim a measured two-tier local-edge service loop.

Experiments 3-6 may follow Gate S before Gate H. SVDB and NSRDB must never be used to choose architecture, calibration, threshold, quantization, lead profile, or any hardware-failure fallback. If Gate H fails for the software-frozen L2 model, either report the deployment mismatch transparently or start the predeclared L1 fallback from development data and a new Gate S manifest; do not select using observed external-test performance.

## 3. Current Project Status

| Component | Status | Evidence | Interpretation |
|---|---|---|---|
| MIT-BIH record-level baseline | Complete | `models/pvc_calibrated_metrics.json` | Historical single-database baseline |
| Waveform + pre-RR baseline | Complete | `models/pvc_rr_calibrated_metrics.json` | Causal rhythm-feature comparison |
| Waveform + pre/post-RR baseline | Complete | `models/pvc_prepost_rr_calibrated_metrics.json` | Delayed-decision upper comparison, not preferred edge model |
| Frozen INCART evaluation | Complete, historical | `models/incart_external_frozen_metrics.json` | Cannot remain final external evidence after INCART enters training |
| Automatic R-peak exploratory evaluation | Complete, exploratory | `models/streaming_rpeak_*_metrics.json` | Shows pipeline degradation; must be repeated with the final model |
| Multi-source audit and preprocessing contract | Complete | `results/experiment0/` | Experiment 0 passed |
| Advisor-approved deployment architecture | Conceptual boundary frozen; exact endpoint, transport, and gateway identities pending | AGENTS.md Section 8 | ECG acquisition endpoint plus local edge gateway; cloud-independent critical loop |
| Multi-source model training | Complete | `results/experiment1/runs.csv`, `results/experiment1/summary.json`, `results/experiment1/per_record_predictions/` | M0/M1/M2 each completed with five outer folds and three fixed seeds; development evidence only. |
| Single- versus two-lead comparison | Complete; L2 is the software-development lead profile pending deployment evidence | `results/experiment2/summary.json`, `results/experiment2/completion.json`, `results/experiment2/runs.csv`, `results/experiment2/per_record_predictions/` | L1/L2 each completed with five outer folds and three fixed seeds. L2 may enter Gate S after final-architecture revalidation; its endpoint acquisition, transport bandwidth, and gateway resource suitability remain a Gate H decision. |
| Pre-freeze model optimization | Experiment 2.5 development decision recorded; software-freeze evidence remains pending | `results/experiment2_5/` | O1 is the preregistered control for the one bounded Experiment 2.6 refinement. O2 had higher paired PVC-record F1 but increased PVC-free-record false positives, violating the safety gate. SVDB and NSRDB remain inaccessible. |
| Experiment 2.6 controlled refinement | Complete; P0 selected, therefore O1 retained | `results/experiment2_6/protocol.json`, `results/experiment2_6/runs.csv`, `results/experiment2_6/summary.json` | P1 did not improve paired PVC-record F1 and slightly increased PVC-free false decisions; P2/P3 failed the Stage A PVC-free safety gate. No additional architecture search is permitted. |
| Experiment 2.6R Recall-stability revision | Complete; no candidate advanced beyond Stage A | `results/experiment2_6r/` | R1 violated the PVC-free safety gate; R2 met recall and safety but failed F1 non-inferiority; R3 found no feasible inner-fold safety threshold and reproduced P0. P0/O1 remains retained and model search stops. |
| Final-architecture M0/M1/M2 and L1/L2 revalidation | Complete; M1/L2 retained | `results/final_revalidation/summary.json`, `results/final_revalidation/runs.csv` | 60 runs completed with five record-level folds and three fixed seeds. M1/L2 reproduces the P0/O1 metrics; M1 versus M0 paired F1 CI crosses zero; M2 increases recall but materially reduces F1 and increases PVC-free false decisions; L2 improves paired F1 over L1 with a positive bootstrap lower bound. No external data were accessed. |
| Gate S software-freeze attempts | Conversion-only compatibility checks complete; Gate S remains closed | `results/software_freeze_gate_s_retry6/`, `results/gate_s_qat_retry2/`, `results/gate_s_converter_tf216/`, `results/gate_s_converter_tf215/`, `results/gate_s_converter_tf215_rebuild_calibrated/` | The original full-int8 run failed p99 0.163 (extreme-coverage preflight 0.111); fixed-range fake QAT reduced p99 to 0.0958 but did not pass. TensorFlow 2.16 failed internally during conversion; TensorFlow 2.15 converted the unchanged source weights but p99 was 0.1552. All failed artifacts are preserved, no external data were accessed, and no model/threshold choice changed. |
| Gate S standard QAT remediation | Complete; failed multiple immutable equivalence criteria | `results/gate_s_tfmot_qat/protocol.json`, `results/gate_s_tfmot_qat_run3/manifest.json`, `results/gate_s_tfmot_qat_run3/int8_equivalence.json` | The one registered standard TFMOT QAT attempt retained 3,506 trainable parameters and used only M1 development records, five folds, seed 20260803, 19 epochs, 150 steps/epoch, and OOF-only calibration/threshold selection. Its int8 artifact was 11,000 bytes, but agreement was 0.983, mean error 0.0219, p99 error 0.2471, and F1 loss 0.0232; only size and Recall-loss criteria passed. The stopping rule is reached: retain P0/O1, stop all model/quantization searches, and keep external evaluation locked. |
| Gate S2 direct int8 calibration amendment | Complete; passed under the explicitly selected TensorFlow 2.15.1 runtime | `results/gate_s2_int8/protocol.json`, `results/gate_s2_int8/execution_failure.json`, `results/gate_s2_int8_tf215_run1/int8_direct_evaluation.json`, `results/gate_s2_int8_tf215_run1/manifest.json` | The TensorFlow 2.16.1 run remains an aborted toolchain record. The one fixed TensorFlow 2.15.1 rerun completed five int8 OOF folds and a final int8 artifact, using MIT-BIH + INCART only. Support-aware macro F1 was 0.8335, recall 0.9075, precision 0.8336, record-macro AUPRC 0.8969, Brier 0.0236, and PVC-free false decisions 83 versus P0 control 87; artifact size was 11,240 bytes and held float/int8 decision agreement was 1.000. All registered Gate S2 criteria passed. The original Gate S p99 failure remains disclosed as a descriptive paired diagnostic (Gate S2 p99 0.0792) and is not reclassified. |
| Frozen SVDB evaluation | Complete; frozen int8 P0/O1 evaluated once on SVDB | `results/experiment3/protocol.json`, `results/experiment3/summary.json`, `results/experiment3/per_record_metrics.csv`, `results/experiment3/integrity_verification.json` | Under the passed Gate S2 artifact, fixed calibration, and threshold `0.49`, the oracle-annotation SVDB evaluation yielded support-aware record-macro PVC F1 `0.6736` (record-bootstrap 95% CI `0.5945` to `0.7493`), recall `0.7154`, precision `0.7411`, record-macro AUPRC `0.8087`, Brier `0.0323`, and pooled confusion matrix TN/FP/FN/TP `158070/4196/2660/7280`. This is an isolated cross-database holdout, not a pristine blind test; no tuning or selection followed. |
| NSRDB false-alarm evaluation | Complete; frozen int8 P0/O1 evaluated once on NSRDB | `results/experiment4/protocol.json`, `results/experiment4/summary.json`, `results/experiment4/per_record_false_alarms.csv`, `results/experiment4/integrity_verification.json` | At fixed threshold `0.49`, 1,113 false PVC detections occurred over 1,729,496 annotated `N` beats and 437.49 h: false-positive rate `0.000644` and 2.544 false PVC detections/h. Under the existing `policy-v1` three-events-in-30-s aggregation, the overall false service-escalation rate was `0.606/h` (aggregate target <= `1/h`), but the record-bootstrap 95% CI was `0.037` to `1.638/h` and worst record `nsrdb:16272` was `9.0/h`; the target is therefore not uniformly supported across records. Signal-quality and automatic-R-peak relationships remain deferred to Experiment 5. No tuning or selection followed. |
| Final automatic R-peak evaluation | Complete; frozen int8 P0/O1 evaluated with the existing causal detector | `results/experiment5_retry2/protocol.json`, `results/experiment5_retry2/summary.json`, `results/experiment5_retry2/per_record_metrics.csv`, `results/experiment5_retry2/per_record_predictions.npz` | Across 219 records, 218 had eligible N/V classifier windows; `mitdb:232` had none after the fixed boundary exclusion and remains detector-only. At the immutable threshold `0.49`, automatic R-peak detection had sensitivity `0.9522`, PPV `0.9956`, and F1 `0.9734`. Pooled PVC recall fell from `0.8741` with reference peaks to `0.7762` with automatic peaks (F1 `0.8531` to `0.8078`); the PVC-miss decomposition was 3,325 unmatched V peaks, 317 window-mislocalizations, and 4,660 classifier-or-causal-RR errors. No detector, model, calibration, threshold, quantization, preprocessing, lead profile, or seed selection followed. Initial and retry-1 evaluator crashes are preserved under `results/experiment5/` and `results/experiment5_retry1/`; their boundary-case repairs did not change the frozen system. |
| Robustness and fault injection | Not started | Planned Experiments 6 and 8 | Mandatory for service-loop claim |
| Two-tier edge hardware benchmark | Not started; intentionally scheduled after software evaluation and Experiment 8A | Planned Experiment 7 / Gate H | Mandatory for the acquisition-endpoint plus local-gateway claim |
| Software-only local orchestration and feedback prototype | Not started | Planned Experiment 8A | Uses an explicitly simulated replay endpoint and local MQTT v5 transport; cannot support physical acquisition, hardware latency, or energy claims. |
| Declared-hardware local orchestration and feedback confirmation | Not started | Planned Experiment 8B / Gate H | Must close endpoint-to-gateway-to-endpoint without cloud dependency on the declared hardware. |

### Verified Historical Results

These numbers describe previous MIT-BIH-only models. They are baselines, not final paper results.

| Condition | PVC precision | PVC recall | PVC F1 | Specificity |
|---|---:|---:|---:|---:|
| Waveform only, nested calibration | 0.896 | 0.842 | 0.868 | 0.991 |
| Waveform + causal pre-RR | 0.887 | 0.894 | 0.890 | 0.989 |
| Waveform + pre/post-RR | 0.913 | 0.913 | 0.913 | 0.992 |
| Frozen pre/post-RR model on INCART | 0.840 | 0.832 | 0.836 | 0.979 |

The pre/post-RR model waits for the following R peak and is therefore unsuitable as the primary low-latency model. It remains an offline upper-comparison condition.

The exploratory streaming experiment showed a major oracle-to-automatic gap. On MIT-BIH, waveform-only PVC F1 fell from 0.898 with reference R peaks to 0.615 with automatic R peaks. On SVDB it fell from 0.416 to 0.295. This supports the need for an end-to-end loop evaluation but does not establish the final system result.

## 4. Research Question Brief

### Primary Research Question

**Under heterogeneous ECG domains and edge-resource constraints, can an ECG acquisition endpoint plus local edge gateway form a multi-source-trained and auditable PVC risk service loop that satisfies predefined clinical-detection and IoT service-quality requirements without cloud-dependent inference?**

### Sub-Questions

1. Does adding INCART and LTDDB to MIT-BIH training improve record-level development performance and frozen cross-database generalization without allowing long records or one source to dominate learning?
2. How much performance is lost when the final classifier is driven by causal automatic R peaks and realistic signal disturbances rather than reference annotations?
3. Can the complete acquisition-endpoint-to-local-gateway-to-endpoint loop meet predefined latency, local-link reliability, false-escalation, resource, and auditability requirements on declared hardware?

All sub-questions inherit the same scope: adult public ECG databases, `N` versus `V` beat classification, fixed 360 Hz canonical gateway timing, fixed one- or two-channel model variants, a two-tier local-edge deployment, and non-diagnostic service escalation.

### FINER Assessment

| Criterion | Score | Rationale |
|---|---:|---|
| Feasible | 4/5 | All five databases are local and audited; model code exists. Hardware and orchestration work remain. |
| Interesting | 4/5 | It addresses the gap between annotation-centered ECG classification and operational connected-health services. |
| Novel | 3/5 | The integrated evidence design may be novel, but literature-level novelty requires a formal search before claiming it. |
| Ethical | 4/5 | Public secondary data and non-diagnostic actions reduce risk; clinical claims and autonomous treatment are excluded. |
| Relevant | 5/5 | Directly matches the CFP's edge, feedback-loop, reliability, and accountability themes. |
| **Average** | **4.0/5** | Proceed, subject to literature and hardware gates. |

## 5. Planned Contribution Claims

These are hypotheses to be tested, not current findings.

1. **Multi-source ECG interpretation:** a source-aware training protocol that standardizes timing and input channels while preserving database and record provenance.
2. **Two-tier local-edge partitioning:** an explicit contract separates acquisition, timestamping, buffering, and transport at the ECG endpoint from interpretation, policy, audit, and acknowledgement at the local gateway.
3. **End-to-end edge evaluation:** a PVC pipeline that measures endpoint acquisition/packetization, local transport, mandatory post-R acquisition delay, automatic R-peak error, gateway inference, policy execution, and acknowledgement separately.
4. **Risk-aware local orchestration and accountable feedback:** the gateway converts calibrated beat-level predictions and signal quality into safe actions while retaining endpoint identity, model hash, threshold version, lead mapping, event identity, policy version, action state, and acknowledgement.
5. **Joint clinical and IoT benchmarking:** evaluation combines PVC metrics with false escalations per hour, endpoint-to-gateway transport reliability, latency decomposition, gateway memory/model size, recovery behavior, and audit completeness.

The paper must not claim novelty from using a 1D-CNN, Pan-Tompkins-style detection, resampling, Platt calibration, MQTT, or TFLite individually. The candidate novelty is the integrated, measurable, and auditable digital service loop and the evidence demonstrating its trade-offs.

## 6. Methodology Blueprint

### Paradigm and Study Type

- Paradigm: pragmatic, quantitative systems research.
- Design: secondary-data machine-learning study plus prototype-based IoT systems benchmarking.
- Primary evidence: public annotated ECG databases and controlled service-loop fault injection.
- Human-subject status: no new human participants are planned. Local institutional requirements must still be checked before manuscript submission.
- Preregistration: recommended before Experiment 1 model selection and mandatory before unlocking final SVDB evaluation if the team wants confirmatory language.

### Units of Analysis

- Model-development split unit: record-level `split_group`, never individual beats.
- Known linked MIT-BIH records `201` and `202`: one split group.
- Clinical metric aggregation: pooled beats and record-level macro results, with the macro result treated as primary for generalization.
- Service metric aggregation: event, session/record, and fault-injection run.
- Hardware metric aggregation: repeated inference or end-to-end loop runs after warm-up.

### Validity Strategy

| Threat | Design Control |
|---|---|
| Beat-level leakage | Split complete records and linked-subject groups only |
| Test-set tuning | Freeze design, weights, calibration, and threshold before final SVDB run |
| LTDDB long-record dominance | Source-balanced, record-balanced, class-aware sampling |
| Database-specific lead semantics | Fixed documented mappings; separate one- and two-channel models |
| Sampling-rate confounding | Canonical 360 Hz across all conditions; no rate selection using SVDB |
| Long-record metric dominance | Report record-level macro metrics and record bootstrap intervals |
| Seed luck | Three fixed training seeds per candidate condition |
| Automatic R-peak coupling | Report oracle and automatic pipelines separately |
| Desktop latency mislabeling | Separate acquisition/packetization, local transport, mandatory post-R wait, gateway preprocessing/R-peak, neural inference, policy, and acknowledgement latency |
| Two-tier role ambiguity | Freeze acquisition-endpoint, local-link, and gateway responsibilities and identifiers before hardware benchmarking |
| Previously observed SVDB | Disclose as isolated external holdout, not pristine blind test |
| Simulated service actions | Label them prototype/fault-injection results, not clinical deployment outcomes |

## 7. Data and Preprocessing Contract

### Data Roles

| Database | Records | Native channels | Native rate | Role | Eligible N | Eligible V |
|---|---:|---|---:|---|---:|---:|
| MIT-BIH Arrhythmia | 48 | 2 | 360 Hz | Training/development | 75,017 | 7,129 |
| INCART | 75 | 12 | 257 Hz | Training/development, selected channels only | 150,346 | 20,006 |
| LTDDB | 7 | 2, one record has 3 | 128 Hz | Long-duration training supplement | 600,224 | 64,094 |
| SVDB | 78 | 2 | 128 Hz | Locked cross-database test | 162,266 | 9,940 |
| NSRDB | 18 | 2 | 128 Hz | Normal-rhythm evaluation only | 1,729,496 | 26 |

Training candidates contain 825,587 eligible `N` and 91,229 eligible `V` windows before balanced sampling.

### Fixed Preprocessing

- Canonical rate: 360 Hz.
- Resampling: polyphase resampling using the audited integer up/down ratios.
- Window: 300 samples centered on the R peak, 150 before and 150 after.
- Window duration: 0.8333 seconds.
- Mandatory post-R acquisition delay: 416.7 ms before detector and inference overhead.
- Labels: `N -> 0`, `V -> 1`; all other WFDB symbols excluded.
- Normalization: zero mean and unit standard deviation independently per channel and per window.
- No imputation or padding at record boundaries.

### Lead Mapping

| Variant | MIT-BIH | INCART | LTDDB/SVDB/NSRDB |
|---|---|---|---|
| One-channel primary | MLII; records 102/104 fall back to channel 0 | II | ECG1 |
| Two-channel ablation | MLII plus the other synchronized channel; 102/104 use both available channels | II + V1 | ECG1 + ECG2 |

The two-channel model means two fixed synchronized channels, not standardized 12-lead fusion. It must be described as a separate model with input shape `(300, 2)`.

### Immutable Audit Evidence

- `results/experiment0/preprocessing_contract.json`
- `results/experiment0/record_manifest.csv`
- `src/multisource_ecg.py`
- `src/07_multisource_data_audit.py`

## 8. System Architecture

### Tier 1: ECG Acquisition Endpoint

- Acquire one or two configured ECG channels at the declared native rate and lead profile.
- Attach endpoint pseudonym, monotonic sample sequence, acquisition timestamp, lead profile, and native sample rate.
- Maintain a bounded ring buffer for temporary local-link interruption and retransmission.
- Send packetized ECG samples or short ordered batches to the local gateway over one frozen transport profile.
- Receive monitor/reacquire acknowledgements and expose their terminal status.
- Do not perform PVC classification, autonomous diagnosis, or unverified online learning.

### Local Endpoint-to-Gateway Transport

- Freeze one primary local transport profile before the final benchmark, such as BLE, local Wi-Fi, or wired LAN/serial.
- Preserve sequence order, detect gaps and duplicates, and expose transport timestamps for latency decomposition.
- Define bounded retry, queue capacity, backpressure, stale-packet, and buffer-overrun behavior.
- The local loop must remain operational without Internet connectivity. Optional remote-review delivery is a separate non-critical egress path.

### Tier 2: Local Edge Gateway

- Reconstruct the configured one- or two-channel ECG stream.
- Apply causal streaming preprocessing and resampling to the 360 Hz canonical gateway timeline.
- Estimate signal quality and detect R peaks causally.
- Extract the fixed post-R window and run the int8 classifier.
- Apply frozen probability calibration, decision threshold, and versioned local service policy.
- Execute monitor/reacquire/buffer/review-request actions, retain the append-only audit record, and acknowledge the acquisition endpoint.
- Emit event summaries rather than continuously forwarding all raw ECG beyond the local system.

### Event Contract

Every accepted gateway event must include at least:

```json
{
  "event_id": "globally unique id",
  "acquisition_endpoint_pseudonym": "non-identifying endpoint id",
  "gateway_pseudonym": "non-identifying gateway id",
  "recorded_at": "UTC timestamp",
  "sample_sequence": "monotonic endpoint sequence or range",
  "transport_profile": "frozen local-link profile",
  "lead_profile": "one_channel or two_channel",
  "native_sample_rate_hz": "declared endpoint rate",
  "gateway_sample_rate_hz": 360,
  "signal_quality": "numeric score and status",
  "r_peak_source": "automatic",
  "pvc_probability": "0..1",
  "decision_threshold": "frozen value",
  "model_sha256": "frozen model hash",
  "policy_version": "local service policy id",
  "endpoint_to_gateway_latency_ms": "measured local-link latency",
  "gateway_decision_latency_ms": "measured gateway processing latency",
  "local_loop_latency_ms": "acquisition through endpoint acknowledgement",
  "requested_action": "monitor, reacquire, buffer_segment, or request_review"
}
```

### Gateway-Local Service Orchestrator

- Validate event schema, sequence continuity, freshness, and idempotency key.
- Reject malformed, stale, duplicate, or hash-mismatched events with retained reason codes.
- Apply a versioned decision policy locally on the gateway.
- Persist the event, action, acknowledgement, and terminal status.
- Coordinate endpoint buffering/retry when the local link is unavailable.
- Queue optional remote-review requests without blocking local monitoring or reacquisition.
- Never convert an isolated model probability directly into a diagnosis.

### Feedback and Validation

- Return acknowledgement from the gateway to the acquisition endpoint for every accepted or rejected local event.
- Record whether the requested action was accepted, completed, rejected, timed out, retried, or failed because the endpoint buffer limit was exceeded.
- For poor signal quality, issue a local reacquisition request to the endpoint.
- For persistent risk events, create a remote-review request according to a frozen aggregation policy, while the local loop continues independently.
- Update operational policy only through an explicit versioned configuration change; do not retrain from service feedback automatically.
## 9. Experimental Program

### Experiment 0: Data Contract and Audit — COMPLETE

Objective: fix data roles, preprocessing, lead mapping, record grouping, and hashes before expanded training.

Success evidence:

- 226 records audited.
- 678 WFDB files hashed.
- All records locally readable.
- Training, external-test, and normal-evaluation roles explicitly separated.

### Experiment 1: Multi-Source Training Ablation - COMPLETE

Objective: determine whether added databases improve generalization under an identical model and protocol.

Conditions:

- `M0`: MIT-BIH only.
- `M1`: MIT-BIH + INCART.
- `M2`: MIT-BIH + INCART + LTDDB.

Primary input: one-channel waveform only. Causal pre-RR is a secondary ablation; post-RR is an offline upper comparison.

Protocol:

1. Freeze source-aware record groups before training.
2. Use five outer record-level development folds. Each fold must contain records from every available training source where feasible.
3. Keep MIT-BIH 201/202 together.
4. Use inner record groups for epoch selection, calibration, and threshold selection; never use an outer test record for these decisions.
5. Sample hierarchically: source, record, then class. Do not concatenate all beats and let LTDDB dominate.
6. Keep architecture, optimizer, epoch policy, augmentation, and threshold rule identical across M0/M1/M2.
7. Run three fixed seeds per condition.
8. Save per-record predictions, calibration traces, configuration, split manifest, and hashes.

Primary development selection rule:

- Require development PVC recall of at least 0.90 if feasible.
- Among feasible operating points, select the highest record-level macro PVC F1.
- If no operating point reaches 0.90 recall, select the highest recall and mark the target as unmet.
- Prefer a larger dataset only if the paired record-level evidence supports it; "more beats" is not itself success.

Outputs:

- `results/experiment1/splits.json`
- `results/experiment1/runs.csv`
- `results/experiment1/per_record_predictions/`
- `results/experiment1/summary.json`
- Candidate models and frozen manifests under `models/experiment1/`

### Experiment 2: One- Versus Two-Channel Ablation - COMPLETE

Objective: determine whether a second synchronized channel justifies its hardware and compute cost.

Conditions:

- `L1`: `(300, 1)` input using the selected training-source condition.
- `L2`: `(300, 2)` input using the same split, seeds, calibration, and threshold protocol.

Compare:

- Record-level macro PVC F1, precision, recall, specificity, AUPRC, and Brier score.
- Parameter count, float model size, int8 model size, peak RAM, and p50/p95 inference latency.
- Per-source and worst-record behavior.

Decision rule: select L2 only if improvement is stable across seeds and records and its local-gateway compute cost plus endpoint acquisition and local-link bandwidth cost remain acceptable. Otherwise retain L1 as the primary system and report L2 as a negative or marginal result.

### Experiment 2.5: Pre-Freeze Model Optimization - COMPLETE

Objective: improve the selected M1/L2 development model to a preregistered, paper-relevant operating point before any final SVDB access, while preserving edge deployability, causal operation, record isolation, and transparent negative-result reporting.

Evidence classification:

- **Verified project evidence:** Experiment 2 completed L1/L2 with five outer folds and three seeds; L2 is development-preferred but is not frozen.
- **Verified literature evidence:** the focused sources listed below support morphology-plus-rhythm modeling, larger temporal receptive fields, compact edge inference, and explicit cross-database evaluation.
- **Targets:** all numerical success thresholds in this section are prospective optimization gates, not achieved findings.
- **Recommendations:** the candidate order and architecture budget are design decisions informed by the evidence; they are not literature-level novelty claims.

Data boundary:

- Use M1 development sources only: MIT-BIH plus INCART.
- Start from L2 `(300, 2)` as the software-development lead profile. Its endpoint acquisition, transport, and gateway resource suitability are deferred to Gate H and may not be inferred from desktop timing.
- Reuse the reviewed Experiment 1/2 record groups and split hash `fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787`.
- Do not enumerate, load, inspect, or derive any tuning information from SVDB or NSRDB.
- Do not use post-RR, whole-record future statistics, future beats, or target-record labels as model inputs.

Metric contract:

- **Support-aware record-macro PVC F1, precision, and recall:** average only over records containing at least one eligible `V` beat, because PVC F1 is mathematically undefined for a record with no positive PVC reference.
- **PVC-free records:** report false PVC decisions, specificity, false PVC detections per hour, and their record distribution separately.
- **Legacy continuity metric:** retain the existing all-record zero-filled macro PVC F1, where undefined record F1 values were recorded as zero. Never replace or hide this value.
- **Probability metrics:** report record-macro AUPRC on records containing both classes and record-macro Brier score on all eligible records.
- **Secondary metrics:** retain pooled precision, recall, F1, specificity, balanced accuracy, AUROC, AUPRC, and Brier score, but do not use pooled accuracy as the selection endpoint.

Prospective optimization targets:

| Metric | Required target |
|---|---:|
| Support-aware record-macro PVC F1 | >= 0.80 |
| Support-aware record-macro PVC recall | >= 0.90 |
| Support-aware record-macro PVC precision | >= 0.75 |
| Record-macro AUPRC | >= 0.88 |
| Record-macro Brier score | <= 0.04 |
| Legacy all-record zero-filled macro PVC F1 | >= 0.70 |
| Three-seed F1 standard deviation | <= 0.04 |
| Paired record-bootstrap F1 difference versus O0 | 95% CI lower bound > 0 |
| PVC-free-record false-positive burden | No increase; preferred reduction >= 25% |
| Full-integer int8 software artifact size | < 1 MiB before Gate S |
| Float/int8 decision agreement | >= 99% before Gate S |
| Declared local-gateway p95 neural inference | < 50 ms at Experiment 7 / Gate H |

Candidate ladder:

- `O0`: corrected-protocol L2 waveform baseline. Reproduce L2 under the Experiment 2.5 epoch-selection, calibration, metric, and artifact rules.
- `O1`: `O0` plus a strictly causal RR branch.
- `O2`: compact multi-scale residual temporal network plus the causal RR branch.
- `O3`: `O2` plus mild training-only robustness augmentation.

Causal RR branch for `O1` and later candidates:

- `pre_rr_seconds`: interval from the previous detected/reference R peak to the current R peak.
- `pre_rr_over_past8_median`: current pre-RR divided by the median of up to eight previously completed RR intervals.
- `pre_rr_over_previous_rr`: current pre-RR divided by the immediately preceding completed RR interval.
- `history_validity`: explicit validity indicators where the available causal history is shorter than requested.
- Fuse the RR vector through an approximately eight-unit MLP after waveform pooling. No feature may depend on the next R peak or a statistic computed from the complete record.

Compact morphology network for `O2`:

- Stem: `Conv1D`, 16 channels, kernel size 7.
- Four depthwise-separable residual temporal blocks with kernel size 9 and dilation rates `1, 2, 4, 8`.
- Use 24 to 32 channels, quantization-compatible activations, and foldable normalization.
- Concatenate global-average and global-max pooled morphology features with the causal RR branch, followed by a 32-unit dense classifier head.
- Target an effective temporal receptive field of at least 127 samples, approximately 353 ms at 360 Hz.
- Hard architecture cap: fewer than 100,000 trainable parameters before any pruning.
- Do not add a Transformer, LSTM, model ensemble, GAN, SMOTE, or a separate derivative branch in the primary ladder.

Training-only robustness for `O3`:

- Apply perturbations only to training windows and record every random seed and generation parameter.
- Candidate perturbations are R-center jitter up to 50 ms, mild baseline wander, weak 50/60 Hz interference, 25-35 dB muscle/noise contamination, and low-probability single-lead dropout.
- Preserve the clean development folds as the selection evidence.
- Experiment 6 must use separately frozen seeds and the declared 20, 10, and 5 dB severity grid; report that the final model was augmentation-trained rather than presenting those noise families as wholly unseen.

Nested protocol:

1. Preserve the five outer record folds and three fixed seeds `20260803`, `20260804`, and `20260805`.
2. Preserve hierarchical source -> record -> class sampling and prevent a source or long record from dominating.
3. Set a maximum of 20-30 epochs before execution and save per-epoch inner-fold curves.
4. Select epoch count only from inner record groups; use the median best epoch across the inner folds for the corresponding outer-fold final fit.
5. Keep optimizer, batch policy, and training budget identical across candidates unless the differing component is the declared ablation.
6. Fit Platt calibration on inner out-of-fold predictions with record/source-balanced weights so that each record and source has controlled aggregate influence.
7. Select one global threshold from inner data only. A feasible threshold must reach both pooled PVC recall >= 0.90 and support-aware record-macro recall >= 0.90.
8. Among feasible thresholds, maximize support-aware record-macro PVC F1; break ties by lower false-positive burden on PVC-free development records.
9. If no threshold is feasible, select the highest support-aware recall and mark the recall target unmet.
10. Save raw and calibrated probabilities, calibration coefficients, threshold traces, per-record metrics, learning curves, configurations, seeds, code hashes, and dependency versions.

Staged execution and stopping rule:

1. **Stage A screening:** run all `O0-O3` candidates on all five folds with the canonical screening seed `20260803`.
2. A candidate advances if it either satisfies every model-quality target or improves support-aware macro F1 by at least 0.02 versus `O0`, loses no more than 0.01 recall, reduces neither source-specific macro F1 by more than 0.01, and passes the software architecture cap. The Stage A `resource_screen_pass` field is only a parameter-count proxy; it is not evidence of int8 conversion or target-hardware performance.
3. **Stage B confirmation:** advancing candidates run the remaining two fixed seeds, yielding five folds by three seeds for every confirmed candidate.
4. Select by the mean across all three seeds and paired per-record evidence. Do not select the best seed.
5. The primary ladder stops after `O3`. If no candidate passes, exactly one preregistered backup is permitted: a training-only source/record-adversarial representation head with no added inference branch.
6. If the backup also fails, record the targets as unmet. Gate B must then contain an explicit proceed/stop decision; do not continue unconstrained architecture search.

Post-selection requirements:

- Complete the one bounded Experiment 2.6 refinement before Gate S; no further model search is permitted after it.
- Revalidate M0/M1/M2 and L1/L2 under the winning architecture and training protocol before claiming that the software-frozen model, rather than the historical baseline, supports those selections.
- Train one canonical final model on all M1 development records with the preregistered seed `20260803` and the median inner-selected epoch count. Fit Platt calibration and one global threshold only from its five-fold development OOF predictions.
- Export full-integer int8 TFLite from development-only representative samples, verify float/int8 agreement, and save weights, calibration parameters, threshold, dependency versions, code hashes, and model hashes under Gate S.
- Freeze one software architecture, source condition, lead profile, seed policy, calibration method, threshold, int8 artifact, and manifest before Experiment 3. Gateway and transport evidence are deferred to Gate H.

Planned outputs:

- `results/experiment2_5/protocol.json`
- `results/experiment2_5/runs.csv`
- `results/experiment2_5/summary.json`
- `results/experiment2_5/calibration_traces/`
- `results/experiment2_5/learning_curves/`
- `results/experiment2_5/per_record_predictions/`
- Candidate exports and manifests under `models/experiment2_5/`

Focused evidence basis and comparability limits:

- de Chazal et al. (2004), DOI <https://doi.org/10.1109/TBME.2004.827359>: independent MIT-BIH record sets; morphology plus heartbeat/RR intervals; VEB sensitivity 77.7% and positive predictivity 81.9%.
- Llamedo et al. (2012), DOI <https://doi.org/10.1109/TITB.2012.2193408>: multilead morphology plus RR features with cross-database corroboration; approximately 90% ventricular sensitivity and positive predictivity, but with up to 12 leads.
- Zhang et al. (2021), DOI <https://doi.org/10.1155/2021/9946596>: inter-patient adversarial CNN plus RR features; strong VEB results on MIT-BIH, but not frozen cross-database evidence.
- Cai et al. (2022), DOI <https://doi.org/10.3390/bios12040185>: MIT-BIH training and INCART testing with automatic R peaks, signal-quality filtering, morphology templates, and rhythm rules; target-record template adaptation makes it non-equivalent to a frozen beat classifier.
- Ivora et al. (2022), DOI <https://doi.org/10.1038/s41598-022-16517-4>: joint automatic QRS detection/classification; external-database classification macro F1 around 0.73, illustrating the operational gap.
- Farag (2023), DOI <https://doi.org/10.3390/s23031365>: tiny inter-patient matched-filter CNN; cross-dataset macro F1 81.24% on INCART, model size about 15 KB, and sub-millisecond mean inference on the declared edge board.
- Rahman et al. (2023), DOI <https://doi.org/10.3390/s23115237>: systematic ECG augmentation survey supporting carefully bounded noise, baseline-wander, powerline, shift, and lead-dropout augmentation while warning that harmful transformations require empirical validation.
- Lim et al. (2024), DOI <https://doi.org/10.1016/j.compbiomed.2024.109062>: adaptive segmentation and relative heart-rate context improve premature-beat recognition; this project adapts the idea to past-only causal RR features.
- Huang et al. (2026), DOI <https://doi.org/10.3389/fphys.2026.1800941>: hierarchical multi-scale residual modeling supports the receptive-field hypothesis, but its approximately 19.1 million parameters and non-inter-patient protocol make its reported F1 values non-comparable to this project.

### Experiment 2.6: Controlled Core-Metric Refinement

Objective: conduct exactly one bounded, development-only refinement after Experiment 2.5 and before Gate S. It may improve support-aware PVC F1, recall, precision, AUPRC, Brier score, PVC-free false-positive burden, and three-seed stability without expanding the model beyond edge suitability.

Data and leakage boundary:

- Use MIT-BIH plus INCART only, the reviewed record groups, and split hash `fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787`.
- Do not enumerate, load, inspect, or derive any tuning information from SVDB or NSRDB.
- Preserve causal operation: no post-RR, future R peaks, whole-record future statistics, or target-record labels as inputs.
- Preserve five outer record folds, linked MIT-BIH 201/202 grouping, source -> record -> class sampling, inner-only epoch selection, calibration, and threshold selection.

Preregistered candidate set:

- `P0`: exact O1 reproduction control.
- `P1`: O1 with robust Platt-calibration and threshold selection from inner OOF predictions.
- `P2`: O1 with bounded causal RR feature clipping and validity-aware normalization.
- `P3`: O1 with fixed mild positive-class loss weighting and RR-branch regularization.
- Do not add a Transformer, LSTM, ensemble, GAN, SMOTE, future-context feature, or unbounded architecture search. Do not expand O3 augmentation severity.

Execution and selection:

1. Stage A runs P0-P3 on all five folds with seed `20260803`.
2. Stage B runs advancing candidates on seeds `20260803`, `20260804`, and `20260805`.
3. A candidate must not increase PVC-free-record false positives over O1, must retain mean support-aware recall >= 0.90, Brier deterioration <= 0.005, three-seed F1 standard deviation <= 0.04, and fewer than 100,000 parameters.
4. Prefer a candidate with three-seed minimum support-aware recall >= 0.90 when feasible. Then rank by paired record-bootstrap F1 versus O1, mean support-aware F1, AUPRC, lower Brier, lower PVC-free burden, and lower parameter count.
5. Claim an F1 improvement only if the 95% paired-bootstrap lower bound exceeds zero. A stability-only candidate may be retained only with F1 non-inferiority lower bound >= -0.01, no PVC-free false-positive increase, and the improved minimum-seed recall.
6. If no candidate is superior under this lexicographic rule, retain O1. The ladder stops after P3; no backup candidate is permitted without an explicit charter revision.

Outputs:

- `results/experiment2_6/protocol.json`
- `results/experiment2_6/runs.csv`
- `results/experiment2_6/summary.json`
- `results/experiment2_6/per_record_predictions/`
- `results/experiment2_6/learning_curves/`
- `models/experiment2_6/`

### Experiment 2.6R: Recall-Stability Protocol Revision

Objective: conduct one explicitly approved, small-scale, development-only protocol revision after Experiment 2.6 and before Gate S. Its primary objective is to improve the minimum of the three seed-level mean support-aware PVC recalls from P0's observed `0.8799` to at least `0.90`; it is not an unconstrained search for a higher mean F1. P0/O1 remains the control and is retained if no R candidate passes every gate.

Evidence and deviation disclosure:

- The immutable Experiment 2.6 artifacts remain the P0 control and must not be overwritten, regenerated, or selected against by best seed.
- R1-R3 use only MIT-BIH plus INCART, the reviewed record groups, split hash `fb003bbd3dc2e4d81f47df732032403a12bc9782a7f76042fdfd59fed346c787`, L2 `(300, 2)` input, causal RR inputs, source -> record -> class sampling, and the fixed seeds `20260803`, `20260804`, and `20260805`.
- SVDB and NSRDB must not be enumerated, loaded, inspected, or used by any 2.6R candidate, calibration, threshold, or selection code path. Before this protocol was registered, the Experiment 0 audit contract was read in full to satisfy the contributor-read requirement and exposed its pre-existing external-database metadata. No waveform, annotation, model, metric, threshold, or candidate information was derived from that exposure. It is an explicitly disclosed read-scope deviation and is not evidence or an input to 2.6R.
- VFDB and CUDB remain excluded because they do not provide reliable beat-level N/V labels for this task.

Preregistered candidate set:

- `R1`: exact P0/O1 architecture and training budget with a batch-level exponential moving average of trainable weights, decay `0.999`, initialized after the first update. Inner validation, epoch selection, final fit, and outer inference use the EMA weights. It adds no inference branch, parameter, or runtime cost.
- `R2`: exact P0/O1 architecture with fixed hard-PVC reweighting only. For each outer fold and seed, construct a four-way cross-fitted P0 proxy cache using only that outer fold's development records. Within each source, mark the bottom `25%` of true PVC beats ranked by their held-out raw P0 probability as hard. During candidate training, apply weight `1.15` only to those marked PVC beats; all other beats retain weight `1.0`. The outer test records are never scored or used to construct the cache.
- `R3`: exact P0/O1 model and P0's uncropped, source- and record-balanced Platt calibration with a `0.01` threshold grid. Unlike P1, which clipped logits and used a `0.02` global grid, R3 chooses only from thresholds that preserve at least `0.90` support-aware recall in every inner validation fold and do not increase the total PVC-free false decisions across those inner folds relative to P0's original aggregate selection. Among feasible thresholds, maximize the minimum inner-fold recall, then support-aware F1, then lower PVC-free burden. If no threshold is feasible, record R3 as infeasible; do not weaken either constraint.

Stage A and Stage B:

1. **Stage A screening:** run R1-R3, each on all five outer folds using only seed `20260803`; P0's existing five corresponding artifacts are read-only controls. The 15 primary R runs are separate from R2's retained development-only proxy-cache fits.
2. A candidate advances only if all five primary runs complete with valid provenance; aggregate PVC-free false decisions do not exceed P0's Stage A total of `161`; mean support-aware recall is at least `0.90`; mean support-aware F1 is no worse than P0 by `0.01`; mean record-macro AUPRC is no worse than P0 by `0.005`; mean record-macro Brier is no worse than P0 by `0.005`; and its model graph hash and parameter count equal P0's `3,506` parameters.
3. **Stage B confirmation:** only Stage A candidates run the remaining two seeds, producing five folds by three seeds. An advancing R2 repeats its cross-fitted cache independently for each seed. Never choose a best seed.
4. A candidate is accepted only if its minimum seed-level mean support-aware recall is at least `0.90`; its aggregate PVC-free false decisions across all 15 runs do not exceed P0's total of `330`; mean F1 is non-inferior to P0 by `0.01` with paired record-bootstrap lower bound at least `-0.01`; mean AUPRC is no worse than P0 by `0.005`; mean Brier is no worse than P0 by `0.005`; three-seed F1 standard deviation is at most `0.04`; and full-integer int8 conversion preflight succeeds from development-only representative data. An F1 improvement may be claimed only if the paired-bootstrap lower bound exceeds zero.
5. Rank accepted candidates by minimum seed-level recall, then paired F1 evidence, mean F1, AUPRC, lower Brier, lower PVC-free burden, and lower parameter count. If no candidate passes, retain P0/O1, record the negative result, and permanently stop model search before Gate S.

Outputs:

- `results/experiment2_6r/protocol.json`
- `results/experiment2_6r/runs.csv`
- `results/experiment2_6r/summary.json`
- `results/experiment2_6r/per_record_predictions/`
- `results/experiment2_6r/learning_curves/`
- `results/experiment2_6r/hard_pvc_proxy_cache/`
- `models/experiment2_6r/` only for a Stage B-accepted candidate's development-only int8 preflight
Stage A outcome — complete negative result:

- `R1` completed five folds with mean support-aware recall `0.8835`, mean F1 `0.7993`, and `303` PVC-free false decisions. It failed the recall, F1, AUPRC, Brier, and PVC-free safety gates.
- `R2` completed five folds with mean support-aware recall `0.9178`, mean F1 `0.8175`, mean AUPRC `0.9197`, mean Brier `0.0245`, and `161` PVC-free false decisions. It met the recall and safety gates but failed the preregistered F1 non-inferiority lower bound of `0.8206` versus P0's Stage A mean.
- `R3` completed five folds; its per-inner-fold safety threshold was infeasible in every fold, so it fell back to the P0 decision rule and reproduced P0's Stage A values: recall `0.9226`, F1 `0.8306`, AUPRC `0.9117`, Brier `0.0251`, and `161` PVC-free false decisions.
- No candidate entered Stage B. The three-seed minimum-recall objective was therefore not tested or achieved. Retain P0/O1 and permanently stop further model, sampling, calibration, and threshold optimization before the required M0/M1/M2 and L1/L2 revalidation.
### Gate S: Software Evaluation Freeze

Gate S occurs after Experiment 2.6 and final-architecture M0/M1/M2 and L1/L2 revalidation, before any final SVDB access. The selected candidate, or O1 if P0-P3 do not improve it, must be trained on all M1 development records with seed `20260803` and the median inner-selected epoch count. Its Platt calibration and global threshold must be fitted from five-fold OOF development predictions only.

Gate S requires a float Keras artifact, a full-integer int8 TFLite artifact, representative quantization samples from development data only, calibration parameters, threshold, model/code/split/dependency hashes, and an immutable manifest. Full-int8 size must be below 1 MiB; float/int8 decision agreement must be at least 99%, mean absolute calibrated-probability error at most 0.02, 99th-percentile absolute error at most 0.05, and support-aware F1 and recall loss at most 0.01 on the held development equivalence set. Gateway latency, memory, endpoint transport, and energy values remain `pending_experiment7` until Gate H.

### Experiment 3: Frozen SVDB Cross-Database Evaluation

Entry gate:

- Experiment 2.5 and the bounded Experiment 2.6 ladder are complete, with all unmet targets retained in the report.
- Gate S records the final development-only model-selection decision and the final-architecture M0/M1/M2 and L1/L2 revalidation status.
- Architecture, training sources, weights, seed policy, calibration, threshold, lead mapping, resampling, float artifact, int8 artifact, and manifest hashes are frozen.
- Declared endpoint, transport, and gateway hardware are not required for Gate S and remain pending Gate H.

SVDB must be evaluated without retraining, threshold changes, quantization changes, lead-profile changes, or selecting among seeds from SVDB performance.

Report:

- Pooled and record-level macro PVC precision, recall, F1, specificity, balanced accuracy, AUROC, AUPRC, and Brier score.
- Confusion matrix and per-record table.
- Record-level bootstrap 95% confidence intervals.
- Worst-decile records and error categories.

Because SVDB was used in prior exploratory work, describe it as an isolated cross-database holdout, not a pristine blind test.

### Experiment 4: NSRDB Normal-Rhythm False-Alarm Evaluation

NSRDB remains completely outside classifier fitting and threshold selection.

Report:

- False-positive rate on annotated `N` beats.
- False PVC detections per hour.
- False service escalations per hour after the frozen aggregation policy.
- Distribution across records and the worst record.
- Relationship between false alarms, signal quality, and R-peak errors.

Engineering target: no more than one false **service escalation** per hour under the frozen aggregation policy. Beat-level false positives remain descriptive and must not be hidden by aggregation.

### Experiment 5: Oracle Versus Automatic R-Peak Pipeline

Repeat the comparison using the final frozen model.

- Oracle path: reference annotation R peaks.
- Operational path: causal automatic R peaks.
- Fixed one-to-one match tolerance: 150 ms.
- Run on development sources, SVDB, and NSRDB without changing the classifier threshold.

Report:

- R-peak sensitivity, positive predictive value, F1, and localization error.
- PVC metrics for oracle and automatic paths.
- Absolute and relative classification degradation.
- Missed PVCs due to missed peaks, mislocalized peaks, and classifier errors.

### Experiment 6: Signal Robustness

Apply perturbations only after model freeze. Do not tune on SVDB perturbation results.

Conditions:

- Baseline wander.
- 50 Hz and 60 Hz interference.
- Band-limited muscle noise at 20, 10, and 5 dB SNR.
- Amplitude scaling from 0.5 to 2.0.
- Short sample loss and timing jitter.
- Lead dropout for the two-channel model.

Report degradation curves for R-peak F1, PVC F1, recall, false escalations per hour, and latency. Every synthetic perturbation must record its generation parameters and random seed.

### Experiment 7: Two-Tier Edge Deployment Benchmark

Experiment 7 is intentionally scheduled after Gate S, Experiments 3-6, and Experiment 8A. It is the first mandatory Gate H measurement. The acquisition endpoint, local transport profile, and local edge gateway must all be declared before Gate H. A Raspberry Pi-class Linux gateway is acceptable for a multi-channel bedside or portable monitor. The acquisition endpoint may initially be a reproducible replay fixture, but physical acquisition claims require a declared ECG front end or acquisition device. An MCU benchmark is optional unless the paper claims MCU or wearable suitability.

Procedure:

1. Record the acquisition endpoint or replay-fixture identity, local transport profile, gateway hardware/OS/runtime, connection parameters, queue capacity, and clock-synchronization method.
2. Export the selected model to full-integer int8 TFLite and verify float-versus-int8 prediction agreement on a held development subset.
3. Warm up the gateway runtime.
4. Run at least 1,000 inference windows on the gateway and at least 30 complete endpoint-to-gateway-to-endpoint sessions.
5. Measure p50, p95, p99, and maximum latency separately for endpoint acquisition/packetization, local transport, mandatory post-R wait, gateway preprocessing/R-peak, neural inference, policy execution, and acknowledgement.
6. Measure model size, gateway peak resident memory, CPU load, endpoint/gateway packet counts, retries, sequence gaps, duplicates, and buffer occupancy.
7. Repeat the local-link screen under the bounded outage and packet-loss conditions later used in Experiment 8.
8. Measure endpoint or gateway energy only if a reproducible instrument is available; otherwise do not estimate it.

Targets:

- Int8 model size below 1 MiB.
- p95 neural inference below 50 ms on the declared local gateway.
- p95 acquisition-to-local-decision latency below 600 ms, including local transport and the 416.7 ms post-R acquisition wait.
- Every accepted session has traceable endpoint, transport, gateway, model, policy, action, and acknowledgement provenance.
- No claim of MCU suitability without an MCU measurement.
- No claim of physical ECG acquisition latency or energy when the endpoint is a replay fixture.

### Experiment 8A: Software-Only Local Digital Service Loop and Fault Injection

Objective: validate the endpoint-to-gateway message contract, local gateway inference and policy implementation, fault recovery, acknowledgement, and auditability before hardware is available. It runs an explicitly simulated replay endpoint, local MQTT v5 transport, and a gateway process on the development computer. It must be labelled software-only simulated service-loop evidence and cannot support physical acquisition, hardware latency, energy, wearable, or deployment claims.

Minimum prototype:

- Explicitly labelled replay endpoint producer with timestamps, sequence numbers, bounded buffer, retransmission, and acknowledgement handling.
- Local MQTT v5 adapter using QoS 1 and idempotency keys; the broker and all critical-path services remain local without Internet dependency.
- Gateway service that validates schema, endpoint identity, freshness, sequence continuity, duplicates, model hash, and policy version.
- Causal stream reconstruction, signal-quality assessment, automatic R-peak interpretation, int8 inference, calibration, and threshold application.
- Versioned `policy-v1`: inadequate signal quality -> `reacquire`; below-threshold valid event -> `monitor`; one above-threshold event -> `buffer_segment`; at least three above-threshold events in 30 seconds -> non-diagnostic `request_review`; duplicates return the original acknowledgement without repeating an action.
- SQLite WAL inbox/outbox audit store with action states, acknowledgement states, and restart recovery. Retry at most three times with 100 ms, 500 ms, and 2 s delays before a retained `failed` terminal status.
- Optional remote-review egress isolated from the latency-critical local loop.

Fault scenarios:

- Duplicate and out-of-order events.
- Temporary local-link outage and endpoint-side buffering.
- 1%, 5%, and 10% packet loss or equivalent injected delivery failure.
- Sequence gaps, duplicates, endpoint buffer saturation, and stale retransmission.
- Delayed acknowledgement.
- Malformed event schema.
- Poor-signal event bursts.
- Model unavailable or hash mismatch.
- Gateway service restart with endpoint- and gateway-queued events.
- Internet outage while the local loop remains available.

Report:

- Endpoint-to-gateway-to-endpoint acknowledgement latency and its component breakdown.
- Successful terminal action rate.
- Duplicate-action rate.
- Event loss after reconnection.
- Recovery time.
- Local-link bandwidth and optional remote-egress bandwidth compared with continuous off-device raw ECG upload.
- Audit completeness: percentage of events with endpoint, transport, gateway, model, policy, action, acknowledgement, and terminal-status provenance.

Service targets:

- Zero duplicate actions under repeated event delivery.
- Zero permanent event loss after a bounded local-link outage and reconnection, within declared endpoint and gateway queue capacities.
- 100% audit-field completeness for accepted events.
- All rejected events retain a reason code.
- Safety fallback is local monitoring or reacquisition, never silent escalation failure.
- Internet loss does not prevent local monitoring, reacquisition, inference, policy execution, acknowledgement, or audit logging.
### Experiment 8B: Declared-Hardware Local Loop Confirmation

Objective: repeat the fixed Experiment 8A scenarios on the Experiment 7 endpoint, transport, and gateway identities. Experiment 8B is required for Gate H and for any claim that the measured two-tier local-edge loop closes without cloud dependency.

All 8A fault scenarios, action semantics, model hash, policy version, queue capacities, and audit fields are frozen before 8B. Hardware-specific measurements must report endpoint-to-gateway-to-endpoint acknowledgement latency, recovery time, duplicate-action rate, permanent event loss after bounded outage, resource use, and audit completeness separately from the software-only simulation.

### Deferred-hardware execution order

`Experiment 2.5 -> Experiment 2.6 -> Gate S -> Experiment 3 -> Experiment 4 -> Experiment 5 -> Experiment 6 -> Experiment 8A -> Experiment 7 -> Gate H -> Experiment 8B`

## 10. Statistical Analysis Plan

- Experiment 2.5 development primary metric: support-aware record-macro PVC F1 at the inner-selected operating point.
- A record contributes to support-aware PVC F1, precision, and recall only when it contains at least one eligible reference `V` beat; do not assign a zero F1 to a mathematically undefined case.
- Preserve and report the legacy all-record zero-filled macro PVC F1 alongside the support-aware metric for continuity with Experiments 1/2.
- For PVC-free records, report specificity, false PVC decisions, false PVC detections per hour, and eventually false service escalations per hour.
- Safety metrics: support-aware PVC recall and false service escalations per hour.
- Probability metrics: record-macro AUPRC and Brier score; AUROC is secondary because of class imbalance.
- Generalization metric: paired per-record difference between candidate models on identical folds.
- Analyze paired F1 differences on PVC-bearing records and false-positive burden separately on PVC-free records.
- Uncertainty: record-level bootstrap with at least 1,000 resamples and a fixed seed.
- Multi-seed reporting: mean, standard deviation, minimum, and maximum across three seeds.
- Do not treat beats from one record as independent participants.
- Do not report only pooled accuracy or pooled F1.
- Do not select a result by the best random seed.
- Disclose that the support-aware/zero-filled distinction was formalized after Experiment 2 but before the frozen external evaluation; never erase the original Experiment 1/2 summaries.
- Report unmet thresholds and negative ablations rather than modifying the protocol post hoc.
- Any new hypothesis after viewing SVDB is exploratory and must be tested on a different future dataset.

## 11. Model and Protocol Freeze Manifest

Before Experiment 3, create one immutable manifest containing:

- Training database names and record IDs.
- Record split groups and split hash.
- Lead profile and fallback records.
- Acquisition endpoint hardware/firmware or replay-fixture identity and simulation status.
- Frozen local transport profile, connection parameters, queue capacities, and clock-synchronization method.
- Local gateway hardware, operating system, runtime, and power mode.
- Source and target sample rates.
- Window and normalization contract.
- Model architecture and parameter count.
- Training seeds and epoch policy.
- Calibration method and coefficients.
- PVC threshold and its selection rule.
- Model SHA-256.
- Dependency versions.
- Source-code commit hash or repository-state identifier.
- Date and operator.

If any field changes, issue a new manifest version. Do not overwrite the frozen version.

## 12. Paper Narrative

### Recommended Working Title

**An Accountable Two-Tier Edge Digital Service Loop for PVC Risk Escalation in Connected Healthcare**

### One-Sentence Positioning

This work studies how a multi-source ECG interpretation model can be embedded in an event-driven and auditable two-tier local-edge service loop formed by an ECG acquisition endpoint and local gateway, and evaluates the complete chain under cross-database, signal, local-link, and resource disturbances.

### Paper Structure

1. Connected-health fragmentation and the need for verifiable service loops.
2. Related work: ECG/PVC edge inference, connected-health orchestration, reliability, and provenance.
3. Proposed ECG acquisition-endpoint plus local-gateway architecture, partition contract, and event schema.
4. Multi-source ECG methodology and leakage controls.
5. Acquisition endpoint, local transport, gateway inference, and service-loop implementation.
6. Clinical, cross-database, resource, and fault-injection results.
7. Discussion: accuracy/latency/reliability trade-offs, limitations, and deployment implications.
8. Conclusion focused on measurable service behavior, not only classifier accuracy.

### Required Figures

1. Two-tier endpoint-to-gateway sensing, interpretation, decision, action, acknowledgement, and feedback architecture.
2. Local service-loop state machine including endpoint buffering, retry, gateway reacquisition acknowledgement, and optional human-review branches.
3. Data-role and leakage-control diagram.
4. M0/M1/M2 and L1/L2 development results with confidence intervals.
5. SVDB/NSRDB oracle-versus-automatic performance.
6. Endpoint acquisition/packetization, local transport, post-R wait, gateway processing/inference/policy, and acknowledgement latency decomposition with hardware resource profiles.
7. Robustness and service fault-recovery curves.

### Required Tables

1. Dataset, sampling-rate, channel, role, and label summary.
2. Model variants and parameter/resource costs.
3. Frozen external-test clinical metrics.
4. Endpoint-to-gateway reliability, false escalation, latency, local-loop recovery, and auditability metrics.
5. Limitations and corresponding mitigations.

## 13. Literature Work Required Before Novelty Claims

Conduct a focused literature review in four streams:

1. Edge or TinyML ECG/PVC detection with cross-database validation.
2. Automatic R-peak plus downstream arrhythmia classification pipelines.
3. Two-tier edge/gateway digital service loops and event-driven local orchestration in connected healthcare.
4. Provenance, reliability, fault recovery, and human-in-the-loop escalation for medical IoT.

For each candidate paper, extract WHY, HOW, WHAT, data, hardware, loop stages implemented, metrics, and unresolved gaps. Do not claim "first," "novel," or "state of the art" until the search and citation verification are complete.

## 14. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Multi-source training does not improve SVDB | High | Report negative result; choose development-selected model; retain service-loop contribution |
| LTDDB dominates due to long recordings | High | Hierarchical source/record/class sampling and per-source reporting |
| Two-channel semantics differ across databases | Medium | Fixed documented mappings; call it two-channel, not uniform lead-space fusion |
| Upsampling 128 Hz appears misleading | Medium | State that it standardizes timing and adds no physiological information |
| Automatic R-peak remains weak | High | Error decomposition; improve detector using development data only; preserve frozen classifier comparison |
| Centered window imposes 416.7 ms delay | Medium | Report explicitly; test a causal shorter-post-window variant only as a preregistered ablation |
| No declared acquisition endpoint, local transport, or gateway in time | Critical | Freeze all three deployment profiles before model freeze; label replay endpoints as simulated |
| Service prototype remains a message demo | High | Demonstrate endpoint buffering, local transport faults, gateway policy execution, acknowledgement, recovery, and audit metrics |
| Endpoint and gateway clocks are not comparable | High | Freeze a clock-synchronization or timestamp-correlation method and report its uncertainty |
| Cloud scope returns through implementation convenience | Medium | Keep cloud inference/orchestration outside the critical loop; isolate optional remote-review egress and test local operation during Internet outage |
| CFP mismatch from classifier-centric writing | Critical | Make service loop the title, architecture, evaluation unit, and discussion center |
| SVDB previously viewed | Medium | Disclose; freeze protocol and avoid "blind test" wording |
| Public databases do not establish clinical benefit | High | Limit claims to technical detection and service feasibility |
| Optimization overfits the reused development folds | High | Predeclare O0-O3, use one screening rule, confirm with all three seeds, allow only one backup, and rely on frozen SVDB for external evidence |
| Metric definition appears changed to improve scores | High | Report support-aware and legacy zero-filled macro F1 together; disclose the pre-external timing and evaluate PVC-free records with false-alarm metrics |
| Optimized architecture reverses M1 or L2 conclusions | High | Revalidate M0/M1/M2 and L1/L2 under the winning protocol before final selection claims |
| Schedule compression | High | Use hard weekly gates and stop low-value architecture exploration |

## 15. Timeline to Submission

| Date | Gate | Required Deliverable |
|---|---|---|
| 2-3 Aug 2026 | Development evidence | Experiments 0-2 artifacts reviewed; M1 selected and L2 development-preferred pending hardware evidence |
| 4-15 Aug | Experiment 2.5 optimization | O0-O3 Stage A, advancing-candidate Stage B, support-aware metrics, and int8 resource screening |
| 16-23 Aug | Final ablation validation | COMPLETE: P0/O1 M0/M1/M2 and L1/L2 revalidation; declared endpoint/local-link/gateway screen remains pending |
| 24-27 Aug | Model-selection gate | Experiment 2.5 decision, all unmet targets retained, final source/lead/seed policy selected |
| 28-31 Aug | Model freeze | COMPLETE: original Gate S attempts failed their immutable pointwise p99 criterion; the explicitly versioned Gate S2 deployment-probability amendment passed under TensorFlow 2.15.1. P0/O1/M1/L2 and the Gate S2 int8 artifact are software-frozen; external evaluation is unlocked. |
| 1-8 Sep | External evaluation | Experiments 3 (SVDB) and 4 (NSRDB) reports complete |
| 9-16 Sep | Operational ECG pipeline | COMPLETE: final automatic R-peak report; Experiment 6 robustness remains pending |
| 17-27 Sep | Two-tier edge deployment | Int8 gateway model plus acquisition-endpoint, local-link, gateway, and acknowledgement benchmark |
| 28 Sep-7 Oct | Service loop | Orchestrator, acknowledgement, audit, and fault-injection results |
| 8-14 Oct | Literature and synthesis freeze | Verified related-work matrix and finalized contribution claims |
| 15-21 Oct | Manuscript draft | Complete figures, tables, and first draft |
| 22-26 Oct | Internal review | Methodology, integrity, CFP-scope, and reproducibility review |
| 27-29 Oct | Revision | Final manuscript and supplementary artifacts |
| 30 Oct | Submission freeze | No new experiments; package verification only |
| 31 Oct | Deadline | Submit through IEEE Author Portal |

## 16. Decision Gates

### Gate A: Experimental Design Freeze

Must pass before Experiment 1:

- Exact record groups saved.
- Hierarchical sampling specified.
- All metrics and thresholds declared.
- Three seeds fixed.
- SVDB and NSRDB access prohibited from training code paths.

### Gate B: Model Selection

Use development evidence only. Do not inspect SVDB to choose M0/M1/M2, L1/L2, architecture, threshold, or seed.

- Execute the preregistered Experiment 2.5 ladder and stopping rule without adding undeclared candidates.
- Select candidates by three-seed means and paired record evidence, never by the best seed.
- Complete the final-architecture M0/M1/M2 and L1/L2 revalidation before claiming those choices apply to the final model.
- If Experiment 2.5 targets remain unmet after the one permitted backup, record an explicit proceed/stop rationale without weakening the targets.

### Gate C: External Test Unlock

Requires a completed Experiment 2.5 decision, a complete freeze manifest, and successful hash verification. External results are write-once evidence; subsequent changes create a new explicitly exploratory version.

### Gate D: CFP Alignment

The prototype must demonstrate every loop stage across the approved local architecture: sensing at the ECG acquisition endpoint; interpretation, decision, policy, and audit at the local gateway; execution and acknowledgement back to the endpoint; and outcome validation. Internet or cloud availability must not be required to close the loop. A model-only or gateway-only message demo fails this gate.

### Gate E: Manuscript Integrity

- Every numerical claim traces to an artifact.
- Every literature claim has a verified source.
- Simulated service behavior is labeled simulated.
- No clinical efficacy or regulatory claim exceeds the evidence.

### Gate S2: Deployment-Probability Protocol Amendment (2026-08-08)

The original Gate S remains a completed negative result and is not reclassified. A
single, explicitly versioned Gate S2 amendment is permitted because the deployed
artifact is full-integer int8, while the original p99 criterion measures pointwise
agreement with a float reference that is not deployed. This amendment is informed
by Jacob et al. (CVPR 2018, DOI <https://doi.org/10.1109/CVPR.2018.00286>), which
defines integer-only inference as a distinct deployment computation, and Guo et al.
(ICML 2017, arXiv:1706.04599), which supports fitting a post-processing calibration
map on held-out predictions. Recent ECG edge studies likewise evaluate quantized
models directly on the target runtime (Hizem et al., 2025, DOI
<https://doi.org/10.3390/s25082496>), while warning that accuracy/resource trade-offs
must be measured together.

Gate S2 is a protocol amendment, not a pass of the original Gate S. It must:

- preserve P0/O1, M1 (MIT-BIH + INCART), L2, the reviewed record groups, causal inputs,
  seed `20260803`, and the frozen epoch policy;
- use only development records and five outer-fold int8 OOF predictions to fit one
  source/record-balanced Platt map and one global threshold for the deployed int8
  artifact; no equivalence-set labels may fit calibration or threshold;
- retain the original float/int8 decision agreement, mean error, and p99 error as
  descriptive paired diagnostics, with the p99 failure disclosed; these values must
  not be silently removed or renamed;
- accept only if the directly calibrated int8 OOF/development evidence satisfies
  support-aware PVC F1 >= 0.80, support-aware PVC recall >= 0.90, support-aware PVC
  precision >= 0.75, record-macro AUPRC >= 0.88, record-macro Brier <= 0.04, legacy
  zero-filled macro F1 >= 0.70, PVC-free false decisions no greater than the P0
  control, int8 size < 1 MiB, and float/int8 decision agreement >= 0.99 on the held
  development equivalence set;
- stop after this one fixed evaluation. It may not search calibration families,
  thresholds, representative sets, converters, architectures, weights, or seeds.

If Gate S2 fails, retain P0/O1 as a development model and keep SVDB/NSRDB locked.
Only a passed Gate S2 may unlock the existing external-evaluation queue, and the
paper must report both the original Gate S failure and the Gate S2 decision.

## 17. Immediate Execution Queue

1. **Complete:** Revalidate M0/M1/M2 and L1/L2 under the retained O1/P0 final protocol; evidence is in `results/final_revalidation/`.
2. Original Gate S conversion attempts and the one registered standard QAT remediation failed immutable equivalence criteria. Preserve those artifacts and do not alter their status.
3. **Complete: Gate S2 (one-time amendment).** The TensorFlow 2.16.1 execution aborted before metrics; the explicitly selected TensorFlow 2.15.1 rerun passed all registered Gate S2 criteria. The software-frozen artifacts are `models/gate_s2_int8_tf215_run1/` and `results/gate_s2_int8_tf215_run1/`. Do not run further model, calibration, threshold, representative-data, converter, or quantization searches.
4. **Complete: Experiment 3 frozen SVDB evaluation.** The passed Gate S2 full-int8 P0/O1/M1/L2 artifact, fixed Platt calibration, and threshold `0.49` were evaluated exactly once on all 78 SVDB records with oracle annotation windows. The result, per-record predictions, record-level bootstrap intervals, and integrity verification are in `results/experiment3/`. No retraining, calibration, threshold, quantization, lead-profile, preprocessing, seed, or model selection followed the result.
5. **Complete: Experiment 4 NSRDB false-alarm evaluation.** The same passed Gate S2 full-int8 P0/O1/M1/L2 artifact, fixed Platt calibration, and threshold `0.49` were evaluated exactly once on all 18 NSRDB records. Overall policy-v1 false service escalation was `0.606/h`, but the record-bootstrap upper confidence limit was `1.638/h` and the worst record was `9.0/h`; preserve this record-level limitation. The result and integrity verification are in `results/experiment4/`. No retraining, calibration, threshold, quantization, lead-profile, preprocessing, seed, or model selection followed the result.
6. **Complete: Experiment 5 automatic R-peak evaluation.** The frozen Gate S2 P0/O1/M1/L2 int8 artifact and threshold `0.49` were evaluated with the existing causal detector. The final report is `results/experiment5_retry2/`; no detector, model, calibration, threshold, preprocessing, lead, quantization, or seed selection followed.
7. **Next gate:** execute Experiment 6 robustness and fault injection under the frozen model, threshold, and policy. Record failures without tuning the frozen classifier or external-test configuration.
8. After Experiment 6, run the software-only Experiment 8A benchmark using the explicitly simulated replay endpoint and local MQTT v5 transport. Do not present 8A as physical deployment evidence.
9. Freeze the declared endpoint, local transport, gateway, queue, and clock method before Experiment 7; repeat the fixed 8A scenarios as Experiment 8B on declared hardware for Gate H.
10. Preserve the current `models/` contents as historical baselines until replacement artifacts are complete and verified.
## 18. Rules for Future Agents and Contributors

### Must

- Read this file and `results/experiment0/preprocessing_contract.json` before changing experiments.
- Preserve database roles and record-group isolation.
- Use `apply_patch` for manual source edits.
- Store configurations, per-record outputs, seeds, and hashes with every run.
- Distinguish verified evidence, inference, target, and recommendation.
- Update the status table and timeline when a gate is completed.
- Keep the service loop central to architecture and evaluation decisions.
- Preserve the advisor-approved ECG acquisition-endpoint plus local-gateway partition and keep the latency-critical loop independent of cloud connectivity.
- Label replay-based acquisition evidence as simulated and separate it from physical acquisition measurements.

### Must Not

- Train on SVDB or NSRDB.
- Tune on final SVDB results.
- Randomly split beats across train and test.
- Count beats as independent patients.
- Report only accuracy or pooled metrics.
- Select the best seed after seeing external-test results.
- Present desktop runtime as device latency.
- Put cloud inference, cloud orchestration, or Internet availability on the latency-critical local-loop path.
- Claim physical acquisition latency, energy, or wearable suitability from a replay endpoint.
- Present synthetic fault injection as a clinical pilot.
- Claim compatibility with arbitrary sample rates or lead counts.
- Claim novelty before the literature audit.
- Delete legacy result artifacts until replacement evidence is verified and archived.

## 19. Completion Definition

The research phase is ready for paper drafting only when all of the following are true:

- Experiment 2.5 is complete under its preregistered ladder, one-backup limit, and stopping rule.
- Final-architecture M0/M1/M2 and L1/L2 comparisons are complete.
- Support-aware and legacy zero-filled record-macro metrics are both preserved, with PVC-free records reported through false-alarm metrics.
- One model and threshold are frozen from development evidence only.
- SVDB and NSRDB evaluations are complete.
- Automatic R-peak and robustness analyses use the final model.
- An int8 model has been measured on the declared local gateway, and the declared acquisition endpoint plus local transport path has been measured or explicitly labelled as a replay-based simulation.
- The event-driven loop closes from ECG acquisition endpoint to local gateway and back to the endpoint with acknowledgement and auditable feedback while Internet connectivity is absent.
- Fault-injection metrics meet or transparently report the predefined targets.
- Literature search supports the final novelty wording.
- Every result cited in the manuscript has a reproducible artifact path.

Until then, the project has a promising and well-audited prototype, not a publication-ready closed-loop healthcare system.


