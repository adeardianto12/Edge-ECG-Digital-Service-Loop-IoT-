"""Versioned, non-diagnostic local service policy."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone

from .contracts import ActionRecord, GatewayEvent, utc_now


class PolicyV1:
    version = "policy-v1"

    def __init__(self, minimum_signal_quality: float = 0.80) -> None:
        self.minimum_signal_quality = minimum_signal_quality
        self._high_risk_times: dict[str, deque[datetime]] = defaultdict(deque)

    def decide(self, event: GatewayEvent) -> ActionRecord:
        if event.signal_quality < self.minimum_signal_quality:
            action, reason = "reacquire", "insufficient_signal_quality"
        elif event.pvc_probability < event.decision_threshold:
            action, reason = "monitor", "below_threshold"
        else:
            timestamp = datetime.fromisoformat(event.recorded_at.replace("Z", "+00:00")).astimezone(timezone.utc)
            window = self._high_risk_times[event.endpoint_id]
            window.append(timestamp)
            while window and (timestamp - window[0]).total_seconds() > 30:
                window.popleft()
            action = "request_review" if len(window) >= 3 else "buffer_segment"
            reason = "persistent_risk_pattern" if action == "request_review" else "single_threshold_crossing"
        return ActionRecord(event.event_id, event.endpoint_id, event.gateway_id, self.version, action, "completed", reason, utc_now())
