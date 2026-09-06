"""
Run the trained model on real field photographs and watch it fall over.

PlantVillage images are single leaves, face up, on a uniform studio background.
PlantDoc images were taken in the field and scraped from the web: whole plants,
soil, sky, hands, shadows, varying focus. Nothing about the disease changed,
only the conditions.

The gap between the two accuracy numbers is the finding of this project.

Caveat, stated plainly: PlantDoc has no healthy-potato class, so this is a
two-class evaluation (early blight vs late blight) against a model trained on
three. A 'healthy' prediction is therefore counted as wrong, which is correct
here - the leaf in front of it is diseased.
"""

import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
PD = ROOT / "data" / "pd"
RESULTS = ROOT / "results"
WEIGHTS = RESULTS / "train" / "weights" / "best.pt"

# PlantDoc folder name -> our class name
FOLDERS = {
    "Potato leaf early blight": "early_blight",
    "Potato leaf late blight": "late_blight",
}
EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def collect() -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for split in ("test", "train"):
        for folder, cls in FOLDERS.items():
            d = PD / split / folder
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix in EXTS:
                    items.append((f, cls))
    return items


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"no trained weights at {WEIGHTS} - run 02_train.py first")

    items = collect()
    if not items:
        raise SystemExit(f"no PlantDoc potato images under {PD}")

    model = YOLO(str(WEIGHTS))
    names = model.names

    correct = 0
    per_class: dict[str, list[int]] = {c: [0, 0] for c in FOLDERS.values()}  # [right, total]
    confusion: Counter[tuple[str, str]] = Counter()

    for path, truth in items:
        pred = model.predict(str(path), imgsz=224, device="cpu", verbose=False)[0]
        got = names[int(pred.probs.top1)]
        confusion[(truth, got)] += 1
        per_class[truth][1] += 1
        if got == truth:
            correct += 1
            per_class[truth][0] += 1

    total = len(items)
    overall = round(correct / total * 100, 2)

    print(f"\nfield images evaluated: {total}")
    print(f"overall accuracy: {overall}%\n")
    print(f"{'true class':<14}{'correct':>9}{'total':>7}{'acc %':>8}")
    for cls, (right, n) in per_class.items():
        acc = round(right / n * 100, 2) if n else 0.0
        print(f"{cls:<14}{right:>9}{n:>7}{acc:>8}")

    print("\nconfusion (true -> predicted):")
    for (truth, got), n in sorted(confusion.items(), key=lambda kv: -kv[1]):
        mark = "  " if truth == got else " x"
        print(f"{mark} {truth:<14} -> {got:<14}{n:>5}")

    out = RESULTS / "field_test.json"
    out.write_text(
        json.dumps(
            {
                "images": total,
                "overall_accuracy": overall,
                "per_class": {c: {"correct": r, "total": n} for c, (r, n) in per_class.items()},
                "confusion": {f"{t}->{g}": n for (t, g), n in confusion.items()},
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
