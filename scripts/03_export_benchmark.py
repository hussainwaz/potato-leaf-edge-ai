"""
Export the trained model to ONNX, quantize it to int8, and measure what that
costs: accuracy, file size, and CPU latency.

This is the actual point of the project. Training a leaf classifier is easy;
knowing what you give up to make it run on a device is the engineering.

Latency method: single-threaded ONNX Runtime, 20 warmup iterations, then 200
timed iterations, reporting the MEDIAN and p95 rather than the mean. Single
threaded because the interesting question is what one core of an embedded
board would do, and a laptop's 10 cores flatter the number badly.
"""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "dataset"
RESULTS = ROOT / "results"
WEIGHTS = RESULTS / "train" / "weights" / "best.pt"

WARMUP = 20
TIMED = 200


def measure_latency(onnx_path: Path, imgsz: int = 224) -> dict[str, float]:
    """Median and p95 single-image, single-thread CPU inference time in ms."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(WARMUP):
        sess.run(None, {name: x})

    samples: list[float] = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 2),
        "p95_ms": round(samples[int(0.95 * len(samples)) - 1], 2),
    }


def accuracy(model_path: Path) -> dict[str, float]:
    """Top-1 on the held-out split, using Ultralytics so preprocessing matches training."""
    metrics = YOLO(str(model_path), task="classify").val(
        data=str(DATASET), imgsz=224, device="cpu", split="val", verbose=False
    )
    return {"top1": round(float(metrics.top1) * 100, 2)}


def mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"no trained weights at {WEIGHTS} - run 02_train.py first")

    # 1. float32 ONNX
    print("exporting float32 ONNX ...")
    fp32 = Path(YOLO(str(WEIGHTS)).export(format="onnx", imgsz=224, simplify=True))

    # 2. int8, dynamic-range weight quantization
    print("quantizing to int8 ...")
    int8 = fp32.with_name("best_int8.onnx")
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    rows = []
    for label, path in (("PyTorch fp32", WEIGHTS), ("ONNX fp32", fp32), ("ONNX int8", int8)):
        print(f"\nevaluating {label} ...")
        row: dict[str, object] = {"model": label, "file": path.name, "size_mb": mb(path)}
        row.update(accuracy(path))
        if path.suffix == ".onnx":
            row.update(measure_latency(path))
        rows.append(row)

    out = RESULTS / "benchmark.json"
    out.write_text(json.dumps(rows, indent=2))

    print(f"\n{'model':<14}{'size MB':>9}{'top-1 %':>9}{'median ms':>11}{'p95 ms':>9}")
    for r in rows:
        print(
            f"{r['model']:<14}{r['size_mb']:>9}{r['top1']:>9}"
            f"{r.get('median_ms', '-'):>11}{r.get('p95_ms', '-'):>9}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
