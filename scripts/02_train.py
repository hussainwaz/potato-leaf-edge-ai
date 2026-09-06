"""
Fine-tune a small ImageNet-pretrained classifier on the potato subset.

Deliberately the smallest sensible model: the point of this project is what
happens when the model has to run on a device, so a big backbone would defeat
the exercise.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "dataset"
RESULTS = ROOT / "results"

# Tried in order; the first one whose weights download wins.
CANDIDATES = ["yolo11n-cls.pt", "yolov8n-cls.pt"]


def pick_model(requested: str | None) -> YOLO:
    names = [requested] if requested else CANDIDATES
    last_error: Exception | None = None
    for name in names:
        try:
            model = YOLO(name)
            print(f"using backbone: {name}")
            return model
        except Exception as exc:  # noqa: BLE001 - report and try the next one
            print(f"  {name} unavailable: {exc}")
            last_error = exc
    raise SystemExit(f"no usable backbone. last error: {last_error}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="backbone checkpoint")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default="mps", help="mps | cpu")
    args = ap.parse_args()

    model = pick_model(args.model)
    model.train(
        data=str(DATASET),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=str(RESULTS),
        name="train",
        exist_ok=True,
        plots=True,
        seed=0,
    )
    print(f"\nweights: {RESULTS / 'train' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
