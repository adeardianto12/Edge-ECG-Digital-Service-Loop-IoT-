"""Run a labelled software-only contract/idempotency demonstration.

This is not an Experiment 8A result.  Full 8A requires a passed Gate S model,
a local MQTT v5 broker, and the preregistered fault-scenario harness.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .contracts import GatewayEvent
from .endpoint_replay import ReplayEndpoint
from .gateway_ingest import GatewayIngest
from .gateway_policy import PolicyV1
from .gateway_store import GatewayStore
from .transport_mqtt import InMemoryTransport


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/experiment8a/contract_demo.json"))
    args = parser.parse_args()
    transport = InMemoryTransport()
    with tempfile.TemporaryDirectory() as directory:
        store = GatewayStore(Path(directory) / "gateway.sqlite")
        policy = PolicyV1()
        model_hash = "simulation-only-model-hash"
        ingest = GatewayIngest("gateway-sim", model_hash, policy.version, store, policy)
        endpoint = ReplayEndpoint("endpoint-sim", "gateway-sim", "two_channel", 360, model_hash, policy.version)
        def event_factory(packet):
            return GatewayEvent(packet.event_id, packet.endpoint_id, packet.gateway_id, packet.sample_sequence, packet.recorded_at, packet.lead_profile, packet.native_sample_rate_hz, packet.transport_profile, packet.model_sha256, packet.policy_version, 0.95, 0.95, 0.80)
        acks = []
        packet = endpoint.packet([[0.0, 0.0]] * 300, event_id="demo-event")
        for payload in (packet.to_dict(), packet.to_dict()):
            ack = ingest.accept(payload, event_factory)
            acks.append(ack.to_dict())
        report = {"evidence_class": "non-benchmark contract demonstration", "simulated_acquisition_endpoint": True, "internet_required": False, "duplicate_returns_original_ack": acks[0] == acks[1], "audit_event_count": store.audit_count(), "acks": acks}
        store.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
