"""Local MQTT v5 adapter and deterministic in-process test transport."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable


class InMemoryTransport:
    """Synchronous local test transport; not evidence of MQTT performance."""

    profile = "simulated_in_memory_transport"

    def __init__(self) -> None:
        self._subscriptions: dict[str, list[Callable[[dict], None]]] = defaultdict(list)

    def subscribe(self, topic: str, callback: Callable[[dict], None]) -> None:
        self._subscriptions[topic].append(callback)

    def publish(self, topic: str, payload: dict) -> None:
        for callback in tuple(self._subscriptions.get(topic, ())):
            callback(payload)


class LocalMqttV5Transport:
    """MQTT v5 QoS 1 adapter; the broker must be local and user-provided."""

    profile = "local_mqtt_v5_qos1"

    def __init__(self, host: str = "127.0.0.1", port: int = 1883) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as error:
            raise RuntimeError("paho-mqtt is required for local MQTT 8A execution") from error
        self._callbacks: dict[str, list[Callable[[dict], None]]] = defaultdict(list)
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
        self._client.on_message = self._on_message
        self._client.connect(host, port, keepalive=30)
        self._client.loop_start()

    def _on_message(self, _client, _userdata, message) -> None:
        for callback in tuple(self._callbacks.get(message.topic, ())):
            callback(json.loads(message.payload.decode("utf-8")))

    def subscribe(self, topic: str, callback: Callable[[dict], None]) -> None:
        self._callbacks[topic].append(callback)
        self._client.subscribe(topic, qos=1)

    def publish(self, topic: str, payload: dict) -> None:
        result = self._client.publish(topic, json.dumps(payload, separators=(",", ":")), qos=1)
        result.wait_for_publish()

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()
