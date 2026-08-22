"""Hardware- and transport-independent service-loop event contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required(payload: dict[str, Any], names: tuple[str, ...]) -> None:
    missing = [name for name in names if payload.get(name) in (None, "")]
    if missing:
        raise ValueError("malformed_schema: missing " + ", ".join(missing))


@dataclass(frozen=True)
class ECGPacket:
    event_id: str
    endpoint_id: str
    gateway_id: str
    sample_sequence: int
    recorded_at: str
    lead_profile: str
    native_sample_rate_hz: float
    transport_profile: str
    model_sha256: str
    policy_version: str
    samples: list[list[float]]
    simulated_acquisition_endpoint: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ECGPacket":
        _required(payload, ("event_id", "endpoint_id", "gateway_id", "recorded_at", "lead_profile", "transport_profile", "model_sha256", "policy_version", "samples"))
        if not isinstance(payload["sample_sequence"], int) or payload["sample_sequence"] < 0:
            raise ValueError("malformed_schema: sample_sequence")
        if payload["lead_profile"] not in {"one_channel", "two_channel"}:
            raise ValueError("malformed_schema: lead_profile")
        if float(payload["native_sample_rate_hz"]) <= 0 or not isinstance(payload["samples"], list):
            raise ValueError("malformed_schema: samples")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GatewayEvent:
    event_id: str
    endpoint_id: str
    gateway_id: str
    sample_sequence: int
    recorded_at: str
    lead_profile: str
    native_sample_rate_hz: float
    transport_profile: str
    model_sha256: str
    policy_version: str
    signal_quality: float
    pvc_probability: float
    decision_threshold: float
    r_peak_source: str = "automatic"
    simulated_acquisition_endpoint: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionRecord:
    event_id: str
    endpoint_id: str
    gateway_id: str
    policy_version: str
    requested_action: str
    status: str
    reason: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Acknowledgement:
    event_id: str
    endpoint_id: str
    gateway_id: str
    ack_status: str
    requested_action: str
    reason: str
    policy_version: str
    acknowledged_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
