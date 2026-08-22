"""Causal packet-to-window preprocessing for the simulation gateway."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.signal import resample_poly

from .contracts import ECGPacket


class CausalSignalProcessor:
    """Forms a latest 300-point window; it never uses a future packet."""

    def __init__(self, target_rate_hz: int = 360) -> None:
        self.target_rate_hz = target_rate_hz
        self._buffers: dict[str, np.ndarray] = defaultdict(lambda: np.empty((0, 0), dtype=np.float32))
        self._previous_peak: dict[str, int] = {}
        self._sample_count: dict[str, int] = defaultdict(int)

    def process(self, packet: ECGPacket) -> tuple[np.ndarray, np.ndarray, float]:
        samples = np.asarray(packet.samples, dtype=np.float32)
        expected_channels = 1 if packet.lead_profile == "one_channel" else 2
        if samples.ndim != 2 or samples.shape[1] != expected_channels or not np.isfinite(samples).all():
            raise ValueError("malformed_schema: samples")
        if packet.native_sample_rate_hz != self.target_rate_hz:
            samples = resample_poly(samples, self.target_rate_hz, int(packet.native_sample_rate_hz), axis=0).astype(np.float32)
        current = self._buffers[packet.endpoint_id]
        if current.size and current.shape[1] != samples.shape[1]:
            raise ValueError("lead_profile_mismatch")
        combined = np.concatenate((current, samples), axis=0) if current.size else samples
        self._buffers[packet.endpoint_id] = combined[-900:]
        self._sample_count[packet.endpoint_id] += len(samples)
        if len(combined) < 300:
            raise ValueError("insufficient_causal_samples")
        window = combined[-300:]
        amplitude = np.ptp(window[:, 0])
        noise = np.median(np.abs(np.diff(window[:, 0]))) + 1e-6
        quality = float(np.clip(amplitude / (20 * noise), 0.0, 1.0))
        local_peak = int(np.argmax(np.abs(window[:, 0] - np.median(window[:, 0]))))
        absolute_peak = self._sample_count[packet.endpoint_id] - 300 + local_peak
        previous = self._previous_peak.get(packet.endpoint_id)
        pre_rr = 1.0 if previous is None else np.clip((absolute_peak - previous) / self.target_rate_hz, 0.3, 2.5)
        self._previous_peak[packet.endpoint_id] = absolute_peak
        rr = np.asarray([pre_rr, 1.0, 1.0, 0.0 if previous is None else 1.0, 0.0], dtype=np.float32)
        mean, std = window.mean(axis=0, keepdims=True), window.std(axis=0, keepdims=True)
        return ((window - mean) / np.maximum(std, 1e-8)).astype(np.float32), rr, quality
