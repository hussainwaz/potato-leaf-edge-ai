"""
Build the potato ImageFolder dataset from PlantVillage, using the dataset's
OFFICIAL leaf-grouped split.

Why the official split matters: PlantVillage contains multiple photographs of
the same physical leaf. A naive random train/test split puts pictures of one
leaf on both sides, so the model is scored partly on leaves it has already
memorised and the accuracy is inflated. The maintainers publish leaf-grouped
split lists precisely to prevent this; we use them.

Input : data/pv/raw/color/Potato___*        (sparse checkout of the dataset)
        data/color_train.txt, color_test.txt (official split lists)
Output: data/dataset/{train,val}/<class>/*.JPG
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = DATA / "pv"
OUT = DATA / "dataset"

# PlantVillage's raw folder names -> the readable class names we train on.
CLASSES = {
    "Potato___Early_blight": "early_blight",
    "Potato___Late_blight": "late_blight",
    "Potato___healthy": "healthy",
}


def read_split(list_file: Path) -> list[str]:
    """Return the potato-only relative paths from one official split list."""
    lines = list_file.read_text().splitlines()
    return [ln.strip() for ln in lines if "/Potato___" in ln]


def build(split_name: str, list_file: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rel in read_split(list_file):
        raw_class = Path(rel).parent.name
        if raw_class not in CLASSES:
            continue
        cls = CLASSES[raw_class]
        src = SRC / rel
        if not src.exists():
            raise FileNotFoundError(f"missing image: {src}")
        dst_dir = OUT / split_name / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)

    # Ultralytics expects train/ and val/; the official "test" split is our val.
    train_counts = build("train", DATA / "color_train.txt")
    val_counts = build("val", DATA / "color_test.txt")

    print(f"{'class':<14}{'train':>8}{'val':>8}{'total':>8}")
    for cls in sorted(set(train_counts) | set(val_counts)):
        t, v = train_counts.get(cls, 0), val_counts.get(cls, 0)
        print(f"{cls:<14}{t:>8}{v:>8}{t + v:>8}")
    tt, vv = sum(train_counts.values()), sum(val_counts.values())
    print(f"{'TOTAL':<14}{tt:>8}{vv:>8}{tt + vv:>8}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
