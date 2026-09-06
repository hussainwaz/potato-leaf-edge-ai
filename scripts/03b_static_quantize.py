"""
Static int8 quantization (QDQ format), with a real calibration set.

Why this exists: script 03 uses quantize_dynamic, which is one line and needs no
data, but it emits ConvInteger operators. Two things go wrong with those:

  1. ONNX Runtime Web's WASM backend has no ConvInteger kernel, so the model
     cannot run in a browser at all.
  2. Activations are quantized at runtime, which on a strong float CPU costs
     more time than the narrower weights save - the 2.2x slowdown measured in
     03.

Static quantization instead measures activation ranges once, ahead of time,
using a calibration sample of real training images, and emits QDQ
(QuantizeLinear / DequantizeLinear) nodes that runtimes have proper kernels
for. This is what you would actually ship to a device.
"""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "dataset"
RESULTS = ROOT / "results"
WEIGHTS = RESULTS / "train" / "weights" / "best.pt"
FP32 = RESULTS / "train" / "weights" / "best.onnx"
OUT = RESULTS / "train" / "weights" / "best_int8_static.onnx"

SIZE = 224
N_CALIB = 200
WARMUP, TIMED = 20, 200


def load_image(path: Path) -> np.ndarray:
    """Resize shortest side to 224, centre crop, scale to [0,1], CHW - as trained."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = SIZE / min(w, h)
    img = img.resize((max(SIZE, round(w * scale)), max(SIZE, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))
    a = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(a, (2, 0, 1))[None, :, :, :]


class Calib(CalibrationDataReader):
    """Feeds a stratified sample of training images so activation ranges are real."""

    def __init__(self, input_name: str) -> None:
        files: list[Path] = []
        for cls_dir in sorted((DATASET / "train").iterdir()):
            if cls_dir.is_dir():
                files.extend(sorted(cls_dir.iterdir())[: N_CALIB // 3])
        self.items = iter([{input_name: load_image(f)} for f in files])
        print(f"calibrating on {len(files)} training images")

    def get_next(self):
        return next(self.items, None)


def latency(path: Path) -> dict[str, float]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, SIZE, SIZE).astype(np.float32)
    for _ in range(WARMUP):
        sess.run(None, {name: x})
    s = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        s.append((time.perf_counter() - t0) * 1000)
    s.sort()
    return {"median_ms": round(statistics.median(s), 2), "p95_ms": round(s[int(0.95 * len(s)) - 1], 2)}


def main() -> None:
    if not FP32.exists():
        raise SystemExit(f"missing {FP32} - run 03_export_benchmark.py first")

    sess = ort.InferenceSession(str(FP32), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    print("quantizing (static, QDQ, per-channel) ...")
    quantize_static(
        str(FP32),
        str(OUT),
        Calib(input_name),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )

    size_mb = round(OUT.stat().st_size / (1024 * 1024), 2)
    print("\nevaluating ...")
    metrics = YOLO(str(OUT), task="classify").val(
        data=str(DATASET), imgsz=SIZE, device="cpu", split="val", verbose=False
    )
    row = {
        "model": "ONNX int8 (static)",
        "file": OUT.name,
        "size_mb": size_mb,
        "top1": round(float(metrics.top1) * 100, 2),
        **latency(OUT),
    }

    path = RESULTS / "benchmark.json"
    rows = json.loads(path.read_text())
    rows = [r for r in rows if r["model"] != row["model"]] + [row]
    path.write_text(json.dumps(rows, indent=2))

    print(f"\n{'model':<20}{'size MB':>9}{'top-1 %':>9}{'median ms':>11}{'p95 ms':>9}")
    for r in rows:
        print(
            f"{r['model']:<20}{r['size_mb']:>9}{r['top1']:>9}"
            f"{r.get('median_ms', '-'):>11}{r.get('p95_ms', '-'):>9}"
        )


if __name__ == "__main__":
    main()
