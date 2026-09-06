# Potato leaf disease on a device: what a benchmark hides

A small edge-AI study. I fine-tuned an image classifier to identify potato leaf
disease, shrank it for on-device deployment, measured what that shrinking cost,
and then ran it on real field photographs.

It scores **100.0%** on the dataset's own held-out split and **57.5%** on real
field photographs of the same two diseases, where guessing scores 50%.

That gap is the point of the project.

---

## Results

### 1. Held-out split, PlantVillage (408 images, 3 classes)

| model | file size | top-1 | median latency | p95 |
|---|---|---|---|---|
| PyTorch fp32 | 3.04 MB | 100.0% | — | — |
| ONNX fp32 | 5.88 MB | 100.0% | 2.78 ms | 2.83 ms |
| ONNX int8, **dynamic** | 1.57 MB | 100.0% | 6.00 ms | 6.05 ms |
| ONNX int8, **static (QDQ)** | **1.70 MB** | 100.0% | **2.07 ms** | 2.12 ms |

Latency is single-image, single-threaded CPU inference on an Apple M4: 20 warmup
iterations, then 200 timed iterations, reported as median and p95 rather than
mean. Single-threaded on purpose — the interesting question is what one core of
an embedded board would do, and a laptop's full core count flatters the number.

**How you quantize matters more than whether you quantize.**

The one-line approach, `quantize_dynamic`, needs no data and is what most
tutorials reach for. It made the model 3.7x smaller and **2.2x slower**.
Dynamic quantization stores weights as int8 but quantizes and dequantizes
activations at runtime; on a CPU with strong float SIMD, that overhead costs
more than the narrower weights save. It also emits `ConvInteger` operators,
which ONNX Runtime Web has no kernel for — so that model cannot run in a
browser at all.

Static quantization measures activation ranges ahead of time from a
calibration sample (200 training images here) and emits QDQ nodes instead. It
is **3.5x smaller than float and 1.34x faster**, and it runs in a browser. It
costs one extra script and a few minutes.

So the honest conclusion is not "int8 is slower". It is that the convenient
form of int8 is slower, and the correct form needs calibration data most
tutorials skip. Accuracy was unchanged at every precision — which mostly tells
you this task is not hard enough to expose quantization damage.

### 2. Real field photographs, PlantDoc (221 images, 2 classes)

| true class | correct | total | accuracy |
|---|---|---|---|
| early blight | 60 | 116 | 51.7% |
| late blight | 67 | 105 | 63.8% |
| **overall** | **127** | **221** | **57.5%** |

Confusion (true → predicted):

| | count |
|---|---|
| early blight → early blight | 60 |
| early blight → late blight | 55 |
| late blight → late blight | 67 |
| late blight → early blight | 36 |
| → healthy (either disease) | 3 |

A two-class coin flip scores 50%. The model scores 57.5%. Nothing about the
diseases changed between the two evaluations — only the photographic
conditions.

---

## Why the model collapses

PlantVillage images are single detached leaves, laid face up, photographed
against a uniform studio background. PlantDoc images were taken in the field:
whole plants, soil, sky, hands, other foliage, uneven light, varying focus.

This is well documented, including by the dataset's own authors:

- **Mohanty, Hughes & Salathé (2016)**, *Using Deep Learning for Image-Based
  Plant Disease Detection*, Front. Plant Sci. 7:1419.
  [doi:10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419) —
  reports 99.35% on the held-out PlantVillage set, falling to **31.40%** on
  images collected under different conditions, and states the work is
  "constrained to the classification of single leaves, facing up, on a
  homogeneous background."
- **Noyan (2022)**, *Uncovering bias in the PlantVillage dataset*,
  [arXiv:2206.04374](https://arxiv.org/abs/2206.04374) — a model given **only 8
  background pixels** reaches 49.0% accuracy against a 2.6% random baseline.
  The backgrounds alone carry enough label-correlated signal to classify by,
  so a network can score well without learning anything about disease.
- **Barbedo (2018)**, *Impact of dataset size and variety…*, Comput. Electron.
  Agric. 153:46-53.
  [doi:10.1016/j.compag.2018.01.009](https://doi.org/10.1016/j.compag.2018.01.009)
  — simple images with simple backgrounds produce classifiers that score well
  academically and underperform on the images practitioners actually take.

My 100% → 57.5% is an independent reproduction of that failure on the potato
classes specifically.

### Avoiding a self-inflicted version of the same problem

PlantVillage contains several photographs of the same physical leaf. A random
`train_test_split` scatters images of one leaf across both sides, so the model
is scored partly on leaves it has already memorised. This project uses the
dataset's **official leaf-grouped split lists** instead, which keep each leaf
wholly on one side. The 100% figure is therefore not the result of that
particular mistake — it is how easy the data genuinely is.

---

## What this means for deployment

If you are building a real in-field diagnostic tool:

1. **Held-out accuracy on PlantVillage tells you almost nothing** about field
   performance. It is a sanity check that the pipeline runs, not evidence the
   model works.
2. **Validate on photographs taken the way users will take them**, in the light
   and framing they will have, at the distance they will use.
3. **Quantize statically, with calibration data.** Dynamic int8 is one line and
   costs 2.2x in latency; static int8 needs a calibration set and pays back
   3.5x in size *and* 1.34x in speed. Measure on the target hardware either
   way — and check the runtime supports the operators your quantizer emits.
4. **The remaining errors are early-vs-late blight confusion** (91 of 94
   mistakes), not disease-vs-healthy. The model can tell something is wrong; it
   cannot reliably tell *which* thing under field conditions. For a tool that
   triages before an agronomist looks, that distinction decides whether it is
   useful.

---

## Reproducing

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/01_build_dataset.py      # build the official split
.venv/bin/python scripts/02_train.py --epochs 20  # ~6 min on an Apple M4 (MPS)
.venv/bin/python scripts/03_export_benchmark.py   # export, dynamic int8, measure
.venv/bin/python scripts/03b_static_quantize.py   # static int8 with calibration
.venv/bin/python scripts/04_field_test.py         # the reality check
.venv/bin/python scripts/05_pick_samples.py       # samples for the web demo
```

### The browser demo

`web/` is a static page that runs the statically-quantized model client-side
with ONNX Runtime Web — the same model, the same preprocessing, on the
visitor's own machine. Serve it with any static server:

```bash
cd web && python3 -m http.server 8899
```

Its JavaScript preprocessing was verified against the Python pipeline: all nine
sample images produce identical labels, with confidence agreeing to within
0.21 percentage points (canvas and PIL resample slightly differently).

Fetching the data (both sparse checkouts pull only the potato folders, ~42 MB
and ~3 MB of images rather than the full 2 GB and 1 GB repositories):

```bash
git clone --filter=blob:none --no-checkout --depth 1 \
  https://github.com/spMohanty/PlantVillage-Dataset.git data/pv
cd data/pv && git sparse-checkout init --cone \
  && git sparse-checkout set raw/color/Potato___Early_blight \
       raw/color/Potato___Late_blight raw/color/Potato___healthy \
  && git checkout && cd ../..

curl -sSL -o data/color_train.txt https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_train.txt
curl -sSL -o data/color_test.txt  https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/splits/color_test.txt

git clone --filter=blob:none --no-checkout --depth 1 \
  https://github.com/pratikkayal/PlantDoc-Dataset.git data/pd
cd data/pd && git checkout HEAD -- \
  "test/Potato leaf early blight" "test/Potato leaf late blight" \
  "train/Potato leaf early blight" "train/Potato leaf late blight"
```

## Setup

- **Model:** YOLO11n-cls, ImageNet-pretrained, 1.53 M parameters, 3.2 GFLOPs
- **Training:** 20 epochs, 224 px, seed 0, Apple M4 via MPS, 6.4 minutes
- **Data:** PlantVillage potato subset, 2,152 images
  (early blight 1,000 / late blight 1,000 / healthy 152), official
  leaf-grouped split = 1,744 train / 408 validation
- **Field set:** PlantDoc potato, 221 images (116 early blight, 105 late blight)

## Honest limitations

- **PlantDoc has no healthy-potato class**, so the field evaluation covers two
  of the three trained classes. A "healthy" prediction on a diseased leaf is
  counted as wrong, which is the right call — but the model's ability to
  recognise a healthy field leaf is untested here.
- **The healthy class is tiny** (152 images, 32 in validation), so its
  contribution to the 100% figure rests on few examples.
- **One hardware target.** Latency was measured on an Apple M4 laptop CPU,
  single-threaded. An ARM embedded board would give different absolute numbers,
  and quite possibly a different sign on the dynamic-int8 result. In the
  browser (WASM) the same model runs at roughly 10-25 ms rather than 2 ms.
- **No hyperparameter search, one seed.** The lab accuracy is saturated, so
  tuning would not have changed the conclusion; the field number would move
  somewhat with a different seed.
- **PlantDoc images were scraped from the web.** The compilation is CC-BY-4.0,
  but individual photograph provenance is mixed.

## Data licences and credit

- **PlantVillage** — Hughes & Salathé, [arXiv:1511.08060](https://arxiv.org/abs/1511.08060);
  Mohanty, Hughes & Salathé 2016. Hugging Face mirror card states CC-BY-SA-3.0.
- **PlantDoc** — Singh, Jain, Jain, Kayal et al., CoDS-COMAD 2020,
  [arXiv:1911.10317](https://arxiv.org/abs/1911.10317). CC-BY-4.0.
- Ultralytics is AGPL-3.0; this repository is coursework, not distributed software.

---

Hussain Nawaz · [hussainnawaz.vercel.app](https://hussainnawaz.vercel.app)
