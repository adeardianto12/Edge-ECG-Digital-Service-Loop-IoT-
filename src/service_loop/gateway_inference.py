"""Gate S int8 inference adapter; it emits probabilities, never diagnoses."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


class FrozenInt8Inference:
    def __init__(self, manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("gate") != "Gate S: Software Evaluation Freeze" or manifest.get("status") != "passed":
            raise ValueError("Gate S manifest is not passed")
        artifact = Path(manifest["artifacts"]["int8_tflite"])
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["artifacts"]["int8_tflite_sha256"]:
            raise ValueError("model_hash_mismatch")
        self.model_sha256 = manifest["artifacts"]["int8_tflite_sha256"]
        self.threshold = float(manifest["calibration"]["threshold"])
        self.coefficient = float(manifest["calibration"]["coefficient"])
        self.intercept = float(manifest["calibration"]["intercept"])
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=str(artifact))
        self.interpreter.allocate_tensors()

    def infer(self, ecg_window: np.ndarray, causal_rr: np.ndarray, signal_quality: float) -> tuple[float, dict]:
        started = time.perf_counter_ns()
        inputs = self.interpreter.get_input_details()
        waveform = next(item for item in inputs if tuple(item["shape"][1:]) == (300, 2))
        rr = next(item for item in inputs if tuple(item["shape"][1:]) == (5,))
        for values, detail in ((ecg_window[None, ...], waveform), (causal_rr[None, ...], rr)):
            scale, zero = detail["quantization"]
            quantized = np.clip(np.rint(values / scale + zero), -128, 127).astype(np.int8)
            self.interpreter.set_tensor(detail["index"], quantized)
        self.interpreter.invoke()
        detail = self.interpreter.get_output_details()[0]
        scale, zero = detail["quantization"]
        raw = float((self.interpreter.get_tensor(detail["index"])[0, 1].astype(np.float32) - zero) * scale)
        logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
        probability = float(1 / (1 + np.exp(-(self.coefficient * logit + self.intercept))))
        return probability, {"model_sha256": self.model_sha256, "decision_threshold": self.threshold, "signal_quality": signal_quality, "inference_latency_ms": (time.perf_counter_ns() - started) / 1e6}
