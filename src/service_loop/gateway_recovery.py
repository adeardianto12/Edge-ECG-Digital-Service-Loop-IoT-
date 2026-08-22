"""Bounded acknowledgement redelivery after a local transport interruption."""

from __future__ import annotations

from .gateway_store import GatewayStore


RETRY_DELAYS_MS = (100, 500, 2000)


class RecoveryManager:
    def __init__(self, store: GatewayStore, transport) -> None:
        self.store, self.transport = store, transport

    def recover(self) -> dict[str, int]:
        delivered = failed = 0
        for row in self.store.pending_outbox():
            try:
                self.transport.publish(row["topic"], __import__("json").loads(row["payload_json"]))
                self.store.mark_delivered(row["event_id"])
                delivered += 1
            except Exception:
                terminal = "failed" if int(row["attempts"]) + 1 >= len(RETRY_DELAYS_MS) else "pending"
                self.store.mark_attempt(row["event_id"], terminal)
                failed += terminal == "failed"
        return {"delivered": delivered, "failed": failed}
