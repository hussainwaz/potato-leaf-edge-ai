"""
Choose the sample images the browser demo ships with, and record what the
model predicts for each so the page can show ground truth honestly.

Field samples are picked to reflect the real hit rate (~57%), not cherry-picked
to fail: we take a mix the model gets right and gets wrong, in roughly the
proportion the full evaluation found.
"""

import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "results" / "train" / "weights" / "best.pt"
LAB = ROOT / "data" / "dataset" / "val"
FIELD = ROOT / "data" / "pd"
OUT = ROOT / "web" / "samples"

FIELD_FOLDERS = {
    "Potato leaf early blight": "early_blight",
    "Potato leaf late blight": "late_blight",
}
N_LAB = 4          # 2 early, 1 late, 1 healthy
N_FIELD_RIGHT = 2  # honest mix at roughly the measured 57%
N_FIELD_WRONG = 3


def predict(model, path: Path) -> tuple[str, float]:
    r = model.predict(str(path), imgsz=224, device="cpu", verbose=False)[0]
    i = int(r.probs.top1)
    return model.names[i], float(r.probs.top1conf)


def main() -> None:
    random.seed(0)
    model = YOLO(str(WEIGHTS))
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "lab").mkdir(parents=True)
    (OUT / "field").mkdir(parents=True)

    manifest: list[dict] = []

    # --- lab samples: whatever the model does, it will be right and certain ---
    wanted = [("early_blight", 2), ("late_blight", 1), ("healthy", 1)]
    for cls, n in wanted:
        files = sorted((LAB / cls).iterdir())
        for src in random.sample(files, n):
            got, conf = predict(model, src)
            dst = OUT / "lab" / f"{cls}__{src.name}"
            shutil.copy2(src, dst)
            manifest.append(
                {
                    "set": "lab",
                    "file": f"samples/lab/{dst.name}",
                    "truth": cls,
                    "pred": got,
                    "conf": round(conf, 4),
                }
            )

    # --- field samples: gather, classify, then take an honest mix ---
    pool: list[tuple[Path, str, str, float]] = []
    for split in ("test", "train"):
        for folder, cls in FIELD_FOLDERS.items():
            d = FIELD / split / folder
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    got, conf = predict(model, f)
                    pool.append((f, cls, got, conf))

    def stratify(items: list, per_class: dict[str, int]) -> list:
        """Take n from each true class so one direction of error cannot dominate."""
        picked = []
        for cls, n in per_class.items():
            group = [it for it in items if it[1] == cls]
            random.shuffle(group)
            picked.extend(group[:n])
        return picked

    right = stratify(
        [p for p in pool if p[1] == p[2]], {"early_blight": 1, "late_blight": 1}
    )
    wrong = stratify(
        [p for p in pool if p[1] != p[2]], {"early_blight": 2, "late_blight": 1}
    )

    for src, truth, got, conf in right + wrong:
        dst = OUT / "field" / f"{truth}__{src.name.replace(' ', '_')}"
        shutil.copy2(src, dst)
        manifest.append(
            {
                "set": "field",
                "file": f"samples/field/{dst.name}",
                "truth": truth,
                "pred": got,
                "conf": round(conf, 4),
            }
        )

    (ROOT / "web" / "samples.json").write_text(json.dumps(manifest, indent=2))
    n_ok = sum(1 for m in manifest if m["truth"] == m["pred"])
    print(f"{len(manifest)} samples, model correct on {n_ok}")
    for m in manifest:
        mark = "ok " if m["truth"] == m["pred"] else "MISS"
        print(f"  {mark} [{m['set']:<5}] {m['truth']:<13} -> {m['pred']:<13} {m['conf']:.3f}")


if __name__ == "__main__":
    main()
