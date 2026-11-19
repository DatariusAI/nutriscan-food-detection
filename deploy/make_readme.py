"""Renders README.md from the run's metrics files (metrics.json, classifier_comparison.csv, pipeline_comparison.csv)."""
import csv, json, sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
m = json.load(open(src / "metrics.json"))
cls_rows = list(csv.DictReader(open(src / "classifier_comparison.csv")))
pipe_rows = list(csv.DictReader(open(src / "pipeline_comparison.csv")))
HF = "https://huggingface.co/DatariusAI/nutriscan-food-models/resolve/main/figures"
SPACE = "https://huggingface.co/spaces/DatariusAI/nutriscan"
ST = sys.argv[2] if len(sys.argv) > 2 else ""

def pct(x): return f"{float(x):.1%}"

cls_table = "\n".join(
    f"| {r['model']} | {pct(r['accuracy'])} | {float(r['macro_F1']):.3f} | {float(r['apple_pie_F1']):.3f} | {float(r['ms_per_image']):.1f} | {float(r['params_M']):.1f} |"
    for r in cls_rows)
pipe_table = "\n".join(f"| {r['approach']} | {pct(r['accuracy'])} |" for r in pipe_rows)
live = f"[Streamlit on Hugging Face]({SPACE}) · [Gradio on Hugging Face]({SPACE}-gradio)" + (f" · [Streamlit Community Cloud]({ST})" if ST else "")

readme = f"""# NutriScan 🥗 — AI Food Detection & Classification

**Snap a meal → see every food item boxed and named → get calories, macros and a healthier swap.**

Capstone project · Deakin Master of Applied AI (Bridge) · Mohammad Alrashed

**Live app:** {live} · **Models:** [DatariusAI/nutriscan-food-models](https://huggingface.co/DatariusAI/nutriscan-food-models) · **Notebook:** [`notebooks/NutriScan_Food_Detection.ipynb`](notebooks/NutriScan_Food_Detection.ipynb)

## Problem
Manual food logging is slow and error-prone. NutriScan needs to:
1. **Detect** food items in a user's photo (where is each item?).
2. **Classify** each item (what is it?).
3. Turn the result into nutrition data and personalised suggestions.

## Data
A 17-class subset of Food-101: apple pie, chocolate cake, donuts, falafel, French fries, hot dog, ice cream, nachos, onion rings,
pancakes, pizza, ravioli, samosa, spring rolls, strawberry shortcake, tacos, waffles. About 16,250 images.
16 classes have ~1,000 images. **Apple pie has 257**, so we oversample it and report macro-F1.
The data has **no bounding boxes**, only class folders.

## Approach
```
photo ──► YOLO11s detector ──► box crops ──► YOLO11s-cls classifier ──► label + confidence ──► nutrition table ──► meal summary
```
| Step | Method |
|---|---|
| Split | Stratified 70 / 15 / 15, seed 42 |
| Classifier A | YOLO11s-cls, ImageNet-pretrained, 224 px, AdamW + cosine LR, apple pie ×3 oversampling |
| Classifier B | EfficientNet-B0 (torchvision), weighted sampler, label smoothing, mixed precision |
| Box labels | Grounding DINO (open-vocabulary detector) prompted with "food." + class name → pseudo-boxes |
| Detector | YOLO11s, COCO-pretrained, 512 px, trained on the pseudo-boxes |
| Pipeline | Detector proposes boxes, classifier re-labels each crop, keep the more confident label |

## Results (held-out test split)
**Classification**

| Model | Accuracy | Macro-F1 | Apple-pie F1 | ms / image (T4) | Params (M) |
|---|---|---|---|---|---|
{cls_table}

**Detection (YOLO11s, scored against pseudo-label boxes)**

| mAP@0.5 | mAP@0.5:0.95 |
|---|---|
| {m['Detector mAP@0.5']} | {m['Detector mAP@0.5:0.95']} |

**Naming the dish end to end (true folder labels)**

| Approach | Accuracy |
|---|---|
{pipe_table}

Whole-image classification scores slightly higher on single-dish photos. The app still ships detect → classify,
because only the boxed version can handle plates with several items and size portions from the box.

![pipeline](docs/figures/fig_pipeline_diagram.png)
![class distribution](docs/figures/fig_class_distribution.png)
![classifier comparison](docs/figures/fig_cls_compare.png)
![confusion matrix](docs/figures/fig_confusion.png)
![pseudo boxes](docs/figures/pseudo_samples.jpg)
![detections](docs/figures/det_samples.jpg)
![per-class AP](docs/figures/fig_det_per_class.png)

## Limitations
- Detector mAP is measured against machine-generated boxes, not hand-drawn ground truth.
- 17 classes only; real meals contain far more foods. An "unknown" threshold is needed before launch.
- Nutrition values are per typical serving (approximate); portion size from one 2-D photo is rough.
- The user-pattern analysis in the notebook uses a simulated food log.

## Repository layout
```
app/app.py                 Streamlit app (also runs on Streamlit Cloud; weights pulled from the HF model repo)
gradio_app/app.py          Gradio version of the app (Hugging Face Space)
app/nutrition.json         per-serving nutrition + healthier swaps for the 17 dishes
notebooks/                 end-to-end notebook with outputs (EDA → training → evaluation → app)
deploy/Dockerfile          Hugging Face Space image (CPU)
deploy/deploy.py           publishes models + Space from Colab
deploy/build_nb.py         generates the notebook from source
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## References
- Bossard, Guillaumin & Van Gool (2014). Food-101 — Mining Discriminative Components with Random Forests. ECCV.
- Tan & Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
- Liu et al. (2023). Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. arXiv:2303.05499.
- Jocher, Qiu & Chaurasia (2024). Ultralytics YOLO11. https://github.com/ultralytics/ultralytics

Nutrition values are approximate and for guidance only, not medical advice.
"""
Path("README.md").write_text(readme)
print("README.md written")
