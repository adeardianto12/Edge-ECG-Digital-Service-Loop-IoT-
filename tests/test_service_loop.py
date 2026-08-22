from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from service_loop.contracts import GatewayEvent
from service_loop.endpoint_replay import ReplayEndpoint
from service_loop.gateway_ingest import GatewayIngest
from service_loop.gateway_policy import PolicyV1
from service_loop.gateway_store import GatewayStore


class ServiceLoopContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = GatewayStore(Path(self.directory.name) / "gateway.sqlite")
        self.policy = PolicyV1()
        self.model_hash = "test-model-hash"
        self.ingest = GatewayIngest("gateway-test", self.model_hash, self.policy.version, self.store, self.policy)
        self.endpoint = ReplayEndpoint("endpoint-test", "gateway-test", "two_channel", 360, self.model_hash, self.policy.version)

    def tearDown(self) -> None:
        self.store.close()
        self.directory.cleanup()

    def event_factory(self, packet):
        return GatewayEvent(
            packet.event_id, packet.endpoint_id, packet.gateway_id, packet.sample_sequence,
            packet.recorded_at, packet.lead_profile, packet.native_sample_rate_hz,
            packet.transport_profile, packet.model_sha256, packet.policy_version,
            0.95, 0.95, 0.80,
        )

    def test_duplicate_event_is_idempotent(self) -> None:
        packet = self.endpoint.packet([[0.0, 0.0]] * 300, event_id="same-event")
        first = self.ingest.accept(packet.to_dict(), self.event_factory)
        second = self.ingest.accept(packet.to_dict(), self.event_factory)
        self.assertEqual(first, second)
        self.assertEqual(self.store.audit_count(), 1)

    def test_sequence_gap_is_rejected(self) -> None:
        first = self.endpoint.packet([[0.0, 0.0]] * 300)
        self.assertEqual(self.ingest.accept(first.to_dict(), self.event_factory).ack_status, "accepted")
        gap = self.endpoint.packet([[0.0, 0.0]] * 300)
        payload = gap.to_dict()
        payload["sample_sequence"] += 1
        self.assertEqual(self.ingest.accept(payload, self.event_factory).reason, "sequence_gap")

    def test_model_hash_mismatch_is_rejected(self) -> None:
        packet = self.endpoint.packet([[0.0, 0.0]] * 300)
        payload = packet.to_dict()
        payload["model_sha256"] = "incorrect"
        self.assertEqual(self.ingest.accept(payload, self.event_factory).reason, "model_hash_mismatch")

    def test_third_risk_event_requests_review(self) -> None:
        actions = []
        for index in range(3):
            packet = self.endpoint.packet([[0.0, 0.0]] * 300, event_id=f"risk-{index}")
            actions.append(self.ingest.accept(packet.to_dict(), self.event_factory).requested_action)
        self.assertEqual(actions, ["buffer_segment", "buffer_segment", "request_review"])


if __name__ == "__main__":
    unittest.main()
