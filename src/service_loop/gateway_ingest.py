"""Gateway schema, provenance, freshness, sequence, and idempotency checks."""

from __future__ import annotations

from datetime import datetime, timezone

from .contracts import Acknowledgement, ECGPacket, utc_now


class GatewayIngest:
    def __init__(self, gateway_id: str, model_sha256: str, policy_version: str, store, policy, stale_after_seconds: float = 10.0) -> None:
        self.gateway_id = gateway_id
        self.model_sha256 = model_sha256
        self.policy_version = policy_version
        self.store = store
        self.policy = policy
        self.stale_after_seconds = stale_after_seconds
        self.next_sequence: dict[str, int] = {}

    def _ack(self, packet: ECGPacket, status: str, action: str, reason: str) -> Acknowledgement:
        return Acknowledgement(packet.event_id, packet.endpoint_id, self.gateway_id, status, action, reason, self.policy_version, utc_now())

    def _reject(self, packet: ECGPacket, reason: str) -> Acknowledgement:
        ack = self._ack(packet, "rejected", "reject", reason)
        self.store.record(packet.event_id, packet.endpoint_id, utc_now(), packet.to_dict(), "rejected", ack.to_dict())
        return ack

    def accept(self, payload: dict, event_factory) -> Acknowledgement:
        try:
            packet = ECGPacket.from_dict(payload)
        except ValueError as error:
            identifier = str(payload.get("event_id", "invalid-event"))
            endpoint = str(payload.get("endpoint_id", "unknown-endpoint"))
            return Acknowledgement(identifier, endpoint, self.gateway_id, "rejected", "reject", str(error), self.policy_version, utc_now())
        existing = self.store.existing_response(packet.event_id)
        if existing is not None:
            return Acknowledgement(**existing)
        if packet.gateway_id != self.gateway_id:
            return self._reject(packet, "gateway_id_mismatch")
        if packet.model_sha256 != self.model_sha256:
            return self._reject(packet, "model_hash_mismatch")
        if packet.policy_version != self.policy_version:
            return self._reject(packet, "policy_version_mismatch")
        try:
            timestamp = datetime.fromisoformat(packet.recorded_at.replace("Z", "+00:00")).astimezone(timezone.utc)
            if (datetime.now(timezone.utc) - timestamp).total_seconds() > self.stale_after_seconds:
                return self._reject(packet, "stale_event")
        except ValueError:
            return self._reject(packet, "malformed_schema: recorded_at")
        expected = self.next_sequence.get(packet.endpoint_id)
        if expected is not None and packet.sample_sequence != expected:
            return self._reject(packet, "duplicate_sequence" if packet.sample_sequence < expected else "sequence_gap")
        try:
            event = event_factory(packet)
        except ValueError as error:
            return self._reject(packet, str(error))
        action = self.policy.decide(event)
        ack = self._ack(packet, "accepted", action.requested_action, action.reason)
        self.store.record(packet.event_id, packet.endpoint_id, utc_now(), {"packet": packet.to_dict(), "event": event.to_dict(), "action": action.to_dict()}, "accepted", ack.to_dict())
        self.next_sequence[packet.endpoint_id] = packet.sample_sequence + len(packet.samples)
        return ack

class LocalGatewayRuntime:
    """Bind ingestion, causal processing, frozen inference, policy, and ACKs."""

    def __init__(self, ingest: GatewayIngest, signal_processor, inference, transport) -> None:
        self.ingest = ingest
        self.signal_processor = signal_processor
        self.inference = inference
        self.transport = transport

    def bind_endpoint(self, endpoint_id: str) -> None:
        self.transport.subscribe(f"ecg/{endpoint_id}/samples", self.handle_packet)

    def handle_packet(self, payload: dict) -> None:
        def event_factory(packet: ECGPacket):
            window, rr, quality = self.signal_processor.process(packet)
            probability, metadata = self.inference.infer(window, rr, quality)
            from .contracts import GatewayEvent
            return GatewayEvent(
                packet.event_id, packet.endpoint_id, packet.gateway_id, packet.sample_sequence,
                packet.recorded_at, packet.lead_profile, packet.native_sample_rate_hz,
                packet.transport_profile, metadata["model_sha256"], packet.policy_version,
                quality, probability, metadata["decision_threshold"],
            )

        ack = self.ingest.accept(payload, event_factory)
        topic = f"ecg/{ack.endpoint_id}/ack"
        self.ingest.store.queue_ack(ack.event_id, topic, ack.to_dict())
        try:
            self.transport.publish(topic, ack.to_dict())
            self.ingest.store.mark_delivered(ack.event_id)
        except Exception:
            pass