"""Export the trained normal-versus-PVC model to full-int8 TFLite and benchmark it."""

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


def representative_dataset(samples: np.ndarray):
    for sample in samples:
        yield [sample[np.newaxis, ...].astype(np.float32)]


def quantize(array: np.ndarray, details: dict) -> np.ndarray:
    scale, zero_point = details["quantization"]
    if scale == 0:
        raise ValueError("TFLite input tensor has no quantization scale")
    dtype = details["dtype"]
    limits = np.iinfo(dtype)
    return np.clip(np.round(array / scale + zero_point), limits.min, limits.max).astype(dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    keras_path = args.models_dir / "tiny_1dcnn_pvc.keras"
    samples_path = args.models_dir / "pvc_representative_samples.npy"
    tflite_path = args.models_dir / "tiny_1dcnn_pvc_int8.tflite"
    if not keras_path.exists() or not samples_path.exists():
        raise FileNotFoundError(
            "PVC model or representative samples are missing; prepare the deployment artifacts before export"
        )

    samples = np.load(samples_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(keras_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_path.write_bytes(converter.convert())

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_data = quantize(samples[0:1], input_details)

    for _ in range(10):
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
    started = time.perf_counter()
    for _ in range(args.runs):
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
    latency_ms = (time.perf_counter() - started) * 1000 / args.runs

    output = interpreter.get_tensor(output_details["index"])
    print(f"TFLite model: {tflite_path}")
    print(f"Size: {tflite_path.stat().st_size / 1024:.1f} KiB")
    print(f"Input: {input_details['shape'].tolist()}, {input_details['dtype']}")
    print(f"Output: {output.tolist()}")
    print(f"Average single-thread latency: {latency_ms:.3f} ms ({args.runs} runs)")


if __name__ == "__main__":
    main()
