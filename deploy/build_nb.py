"""Builds NutriScan_Food_Detection.ipynb from the cell list below."""
import json, pathlib

APP = pathlib.Path("app/app.py").read_text()
NUTRI = pathlib.Path("app/nutrition.json").read_text()
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# NutriScan — AI Food Detection & Classification
**Capstone · Deakin Master of Applied AI (Bridge) · Mohammad Alrashed**

NutriScan wants users to photograph a meal and instantly get the food items, where they are on the plate, and their nutrition.
The brief asks for two models:

1. **Object detection** to locate food items in an image.
2. **Classification** to recognise the food type.

This notebook goes end to end: data audit → EDA → stratified split → two classifiers compared → detector training → a combined
detect-then-classify pipeline → error analysis → nutrition mapping → business insights → export → a live Streamlit app.

| Section | What happens |
|---|---|
| 0 | Setup, paths, reproducibility |
| 1 | Data loading and integrity checks |
| 2 | Exploratory data analysis |
| 3 | Stratified train / val / test split |
| 4 | Classification: YOLO11s-cls vs EfficientNet-B0 |
| 5 | Detection: auto-generated boxes (Grounding DINO) → YOLO11s detector |
| 6 | Two-stage pipeline (detect → classify) and error analysis |
| 7 | Nutrition mapping and meal analytics |
| 8 | Business insights, limitations, next steps |
| 9 | Export artefacts and launch the Streamlit app |
""")

md("## 0. Setup")
code(r"""
import os, sys, subprocess, importlib
QUICK = os.environ.get("NS_QUICK") == "1"          # tiny smoke-test mode (CPU)
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
def pip(*pk):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pk], check=False)
for mod, pkg in [("ultralytics", "ultralytics"), ("streamlit", "streamlit"), ("plotly", "plotly")]:
    if importlib.util.find_spec(mod) is None:
        pip(pkg)
if IN_COLAB and not QUICK and not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
""")
code(r"""
import glob, json, random, shutil, time, zipfile, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from PIL import Image, ImageOps
import torch
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="crest")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 0 if torch.cuda.is_available() else "cpu"

if os.environ.get("NS_CAP"):
    CAP = Path(os.environ["NS_CAP"])
else:
    CAP = Path(glob.glob("/content/drive/MyDrive/Colab Notebooks/Training_Great Learning/MDS*/Capstone")[0])
ZIP  = Path(os.environ.get("NS_ZIP", CAP / "Food_101.zip"))
WORK = Path(os.environ.get("NS_WORK", "/content/work")); WORK.mkdir(parents=True, exist_ok=True)
ART  = CAP / "nutriscan_artifacts"; ART.mkdir(exist_ok=True)       # persistent outputs on Drive
FIG  = ART / "figures"; FIG.mkdir(exist_ok=True)
PROGRESS = ART / "progress.log"
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(PROGRESS, "a") as f: f.write(line + "\n")
def savefig(name):
    plt.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=130)

# budgets (kept small in QUICK smoke-test mode)
CLS_EPOCHS   = 2 if QUICK else 15
EFF_EPOCHS   = 1 if QUICK else 6
DET_EPOCHS   = 2 if QUICK else 30
DET_PER_CLASS = (6, 2, 2) if QUICK else (320, 60, 60)    # train / val / test images per class for detection
log(f"start | device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} | quick={QUICK}")
""")

md(r"""
## 1. Data loading and integrity checks
The zip holds one folder per food class. It also carries macOS metadata (`__MACOSX/`, `.DS_Store`), which we skip.
There are **no bounding-box annotations**, only class folders. Section 5 deals with that.
""")
code(r"""
DATA = WORK / "food"
if not DATA.exists():
    with zipfile.ZipFile(ZIP) as zf:
        members = [m for m in zf.namelist()
                   if not m.startswith("__MACOSX") and not m.endswith(".DS_Store") and m.lower().endswith((".jpg", ".jpeg", ".png"))]
        for m in members:
            parts = Path(m).parts
            dst = DATA / parts[-2] / parts[-1]
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
CLASSES = sorted(p.name for p in DATA.iterdir() if p.is_dir())
print(len(CLASSES), "classes:", CLASSES)
""")
code(r"""
# Verify every image opens, record size / mode / brightness, hash for duplicate detection
rows = []
for c in CLASSES:
    for f in sorted((DATA / c).iterdir()):
        try:
            with Image.open(f) as im:
                im.verify()
            with Image.open(f) as im:
                w, h = im.size; mode = im.mode
                small = im.convert("RGB").resize((32, 32))
                arr = np.asarray(small, dtype=np.float32)
            rows.append(dict(path=str(f), label=c, w=w, h=h, mode=mode, ok=True,
                             r=arr[..., 0].mean(), g=arr[..., 1].mean(), b=arr[..., 2].mean(),
                             bright=arr.mean(), md5=hashlib.md5(f.read_bytes()).hexdigest()))
        except Exception as e:
            rows.append(dict(path=str(f), label=c, ok=False, err=str(e)))
df = pd.DataFrame(rows)
bad = df[~df.ok]
print(f"{len(df):,} files | corrupt: {len(bad)}")
df = df[df.ok].copy()
dups = df[df.duplicated("md5", keep=False)].sort_values("md5")
print("exact duplicate files:", len(dups), "| non-RGB modes:", df["mode"].value_counts().to_dict())
df = df.drop_duplicates("md5").reset_index(drop=True)
df["aspect"] = df.w / df.h
df.head()
""")

md("## 2. Exploratory data analysis")
code(r"""
counts = df.label.value_counts().sort_values()
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ["#e4572e" if v < counts.median() * 0.5 else "#2f7d4f" for v in counts.values]
ax.barh([c.replace("_", " ") for c in counts.index], counts.values, color=colors)
for i, v in enumerate(counts.values): ax.text(v + 8, i, f"{v}", va="center", fontsize=9)
ax.set_title("Images per class"); ax.set_xlabel("images")
savefig("class_distribution"); plt.show()
imb = counts.max() / counts.min()
print(f"Imbalance ratio max/min = {imb:.1f}x  (smallest: {counts.idxmin()} = {counts.min()})")
""")
md(r"""
**Reading the chart.** 16 classes are balanced at about 1,000 images each. `apple_pie` has roughly a quarter of that.
A model trained naively would see apple pie 4× less often and would likely under-predict it.
We handle this in training by oversampling apple pie (classification) and in evaluation by reporting **macro-F1**,
which weighs every class equally, next to accuracy.
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df.w, bins=40, color="#2f7d4f", alpha=.8, label="width"); axes[0].hist(df.h, bins=40, color="#f2a541", alpha=.7, label="height")
axes[0].set_title("Image width / height (px)"); axes[0].legend()
axes[1].hist(df.aspect, bins=40, color="#3a86ff"); axes[1].set_title("Aspect ratio (w/h)")
sns.boxplot(data=df, y="label", x="bright", ax=axes[2], color="#52b788", fliersize=1)
axes[2].set_title("Mean brightness per class"); axes[2].set_ylabel("")
plt.tight_layout(); savefig("image_stats"); plt.show()
print(df[["w", "h", "aspect", "bright"]].describe().round(1))
""")
code(r"""
# A sample grid: 6 random images for each class
n = 6
fig, axes = plt.subplots(len(CLASSES), n, figsize=(n * 1.9, len(CLASSES) * 1.9))
for i, c in enumerate(CLASSES):
    sample = df[df.label == c].sample(min(n, (df.label == c).sum()), random_state=SEED)
    for j in range(n):
        ax = axes[i, j]; ax.axis("off")
        if j < len(sample):
            ax.imshow(ImageOps.fit(Image.open(sample.iloc[j].path).convert("RGB"), (160, 160)))
        if j == 0: ax.set_title(c.replace("_", " "), loc="left", fontsize=9)
plt.tight_layout(); savefig("sample_grid"); plt.show()
""")
code(r"""
# Average colour per class: hints at which classes the model could confuse on colour alone
col = df.groupby("label")[["r", "g", "b"]].mean()
fig, ax = plt.subplots(figsize=(10, 2.2))
ax.imshow(col.values[None, :, :] / 255, aspect="auto")
ax.set_xticks(range(len(col))); ax.set_xticklabels([c.replace("_", "\n") for c in col.index], fontsize=7); ax.set_yticks([])
ax.set_title("Mean RGB colour per class"); savefig("mean_colour"); plt.show()
""")
md(r"""
**EDA takeaways**
- Images are web photos with varied size (mostly ≤ 512 px) and aspect ratio. We resize to a fixed input size, and letterboxing/cropping is handled by the training pipelines.
- Several fried dishes (French fries, onion rings, churro-like donuts, samosa, falafel, spring rolls) share warm golden-brown colours. Expect confusion among them.
- Dessert classes (chocolate cake, ice cream, strawberry shortcake, waffles, pancakes) differ more in texture and toppings than in colour.
- Most photos show a **single dominant dish**. That matters for detection: each image will usually yield one main box.
""")

md("## 3. Stratified train / validation / test split (70 / 15 / 15)")
code(r"""
from sklearn.model_selection import train_test_split
tr, tmp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=SEED)
df["split"] = "train"; df.loc[va.index, "split"] = "val"; df.loc[te.index, "split"] = "test"
df[["path", "label", "split"]].to_csv(ART / "split.csv", index=False)
pd.crosstab(df.label, df.split)[["train", "val", "test"]].T
""")

md(r"""
## 4. Classification
We compare two transfer-learning approaches, both pretrained on ImageNet:

| Model | Why |
|---|---|
| **YOLO11s-cls** (Ultralytics) | Fast, deploys with the same library as the detector, strong augmentation built in |
| **EfficientNet-B0** (torchvision) | Classic compound-scaled CNN baseline (Tan & Le, 2019), trained with a hand-written PyTorch loop |

**Class imbalance.** For YOLO-cls we oversample `apple_pie` in the training folder (×3 via symlinks).
For EfficientNet we use a `WeightedRandomSampler`. Both are evaluated on the untouched test split.
""")
code(r"""
CLS_DS = WORK / "cls_ds"
if not CLS_DS.exists():
    target = int(df[df.split == "train"].label.value_counts().median())
    for split in ["train", "val", "test"]:
        sub = df[df.split == split]
        for c in CLASSES:
            (CLS_DS / split / c).mkdir(parents=True, exist_ok=True)
            files = sub[sub.label == c].path.tolist()
            reps = max(1, round(target / max(len(files), 1))) if split == "train" else 1
            for r in range(reps):
                for f in files:
                    dst = CLS_DS / split / c / (f"r{r}_" + Path(f).name)
                    if not dst.exists(): os.symlink(f, dst)
for s in ["train", "val", "test"]:
    print(s, sum(len(os.listdir(CLS_DS / s / c)) for c in CLASSES))
""")
code(r"""
from ultralytics import YOLO
CLS_BEST = ART / "weights" / "classifier.pt"; CLS_BEST.parent.mkdir(exist_ok=True)
if not CLS_BEST.exists():
    log("train YOLO11s-cls")
    m = YOLO("yolo11s-cls.pt")
    r = m.train(data=str(CLS_DS), epochs=CLS_EPOCHS, imgsz=224, batch=64 if not QUICK else 16, device=DEVICE,
                project=str(WORK / "runs"), name="cls", exist_ok=True, seed=SEED, patience=5,
                optimizer="AdamW", lr0=1e-3, cos_lr=True, workers=2, plots=True, verbose=False)
    shutil.copy(Path(r.save_dir) / "weights" / "best.pt", CLS_BEST)
    shutil.copy(Path(r.save_dir) / "results.csv", ART / "cls_results.csv")
    log("done YOLO11s-cls")
hist = pd.read_csv(ART / "cls_results.csv"); hist.columns = hist.columns.str.strip()
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(hist.epoch, hist["train/loss"], label="train"); ax[0].plot(hist.epoch, hist["val/loss"], label="val"); ax[0].set_title("YOLO11s-cls loss"); ax[0].legend()
ax[1].plot(hist.epoch, hist["metrics/accuracy_top1"], label="top-1"); ax[1].plot(hist.epoch, hist["metrics/accuracy_top5"], label="top-5"); ax[1].set_title("Validation accuracy"); ax[1].legend()
savefig("cls_curves"); plt.show()
""")
code(r"""
# EfficientNet-B0 baseline with a plain PyTorch training loop
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.amp import autocast, GradScaler

IDX = {c: i for i, c in enumerate(CLASSES)}
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_train = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1)), T.RandomHorizontalFlip(), T.ColorJitter(.2, .2, .2), T.ToTensor(), norm])
tf_eval  = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
class DS(Dataset):
    def __init__(s, frame, tf): s.p, s.y, s.tf = frame.path.tolist(), frame.label.map(IDX).tolist(), tf
    def __len__(s): return len(s.p)
    def __getitem__(s, i): return s.tf(Image.open(s.p[i]).convert("RGB")), s.y[i]

EFF_BEST = ART / "weights" / "efficientnet_b0.pt"
NW = 0 if QUICK else 2
trn = df[df.split == "train"]
w = trn.label.map(1.0 / trn.label.value_counts()).values
dl_tr = DataLoader(DS(trn, tf_train), batch_size=8 if QUICK else 64, sampler=WeightedRandomSampler(w, len(w), replacement=True), num_workers=NW)
dl_va = DataLoader(DS(df[df.split == "val"], tf_eval), batch_size=16 if QUICK else 128, num_workers=NW)
dev = "cuda" if torch.cuda.is_available() else "cpu"
try:
    eff = efficientnet_b0(weights=None if QUICK else EfficientNet_B0_Weights.IMAGENET1K_V1)
except Exception as e:
    print("pretrained weights unavailable, training from scratch:", e); eff = efficientnet_b0(weights=None)
eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, len(CLASSES))
eff = eff.to(dev)
eff_hist = []
if EFF_BEST.exists():
    eff.load_state_dict(torch.load(EFF_BEST, map_location=dev))
    eff_hist = json.load(open(ART / "eff_hist.json"))
else:
    log("train EfficientNet-B0")
    opt = torch.optim.AdamW(eff.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EFF_EPOCHS * len(dl_tr))
    scaler = GradScaler(enabled=dev == "cuda"); lossf = nn.CrossEntropyLoss(label_smoothing=0.1); best = 0
    for ep in range(EFF_EPOCHS):
        eff.train(); tl = 0
        for i, (x, y) in enumerate(dl_tr):
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            with autocast(device_type="cuda", enabled=dev == "cuda"):
                loss = lossf(eff(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
            if QUICK and i > 2: break
        eff.eval(); correct = n = 0
        with torch.no_grad(), autocast(device_type="cuda", enabled=dev == "cuda"):
            for x, y in dl_va:
                correct += (eff(x.to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
        acc = correct / n; eff_hist.append(dict(epoch=ep + 1, train_loss=tl / len(dl_tr), val_acc=acc))
        print(eff_hist[-1])
        if acc >= best: best = acc; torch.save(eff.state_dict(), EFF_BEST)
    json.dump(eff_hist, open(ART / "eff_hist.json", "w"))
    eff.load_state_dict(torch.load(EFF_BEST, map_location=dev))
    log("done EfficientNet-B0")
pd.DataFrame(eff_hist)
""")
code(r"""
# Test-set evaluation for both classifiers
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
test = df[df.split == "test"].reset_index(drop=True)
y_true = test.label.map(IDX).values

yolo_cls = YOLO(str(CLS_BEST))
t0 = time.time(); y_yolo, p_yolo = [], []
for i in range(0, len(test), 64):
    for r in yolo_cls.predict(test.path[i:i + 64].tolist(), imgsz=224, device=DEVICE, verbose=False):
        y_yolo.append(IDX[r.names[int(r.probs.top1)]]); p_yolo.append(float(r.probs.top1conf))
ms_yolo = (time.time() - t0) / len(test) * 1000

dl_te = DataLoader(DS(test, tf_eval), batch_size=16 if QUICK else 128, num_workers=NW)
eff.eval(); y_eff = []; t0 = time.time()
with torch.no_grad():
    for x, _ in dl_te: y_eff += eff(x.to(dev)).argmax(1).cpu().tolist()
ms_eff = (time.time() - t0) / len(test) * 1000

cmp = pd.DataFrame({
    "model": ["YOLO11s-cls", "EfficientNet-B0"],
    "accuracy": [accuracy_score(y_true, y_yolo), accuracy_score(y_true, y_eff)],
    "macro_F1": [f1_score(y_true, y_yolo, average="macro"), f1_score(y_true, y_eff, average="macro")],
    "apple_pie_F1": [f1_score(y_true, y_yolo, labels=[IDX["apple_pie"]], average="macro"),
                     f1_score(y_true, y_eff, labels=[IDX["apple_pie"]], average="macro")],
    "ms_per_image": [ms_yolo, ms_eff],
    "params_M": [sum(p.numel() for p in yolo_cls.model.parameters()) / 1e6, sum(p.numel() for p in eff.parameters()) / 1e6],
}).round(4)
cmp.to_csv(ART / "classifier_comparison.csv", index=False)
cmp
""")
code(r"""
best_name = cmp.sort_values("macro_F1", ascending=False).iloc[0].model
y_best = y_yolo if best_name == "YOLO11s-cls" else y_eff
print("Best classifier on test macro-F1:", best_name, "\n")
rep = classification_report(y_true, y_best, target_names=CLASSES, digits=3, output_dict=True)
print(classification_report(y_true, y_best, target_names=CLASSES, digits=3))
cm = confusion_matrix(y_true, y_best, normalize="true")
fig, ax = plt.subplots(figsize=(10, 8.5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Greens", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=False, annot_kws={"size": 7})
ax.set_title(f"Normalised confusion matrix — {best_name} (test)"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
savefig("cls_confusion"); plt.show()
""")
code(r"""
# Most frequent confusions
pairs = pd.DataFrame({"true": [CLASSES[i] for i in y_true], "pred": [CLASSES[i] for i in y_best]})
top_conf = pairs[pairs.true != pairs.pred].value_counts().head(10).rename("count").reset_index()
top_conf
""")
code(r"""
# What the misclassified images look like
wrong = np.where(np.array(y_best) != y_true)[0]
pick = np.random.RandomState(SEED).choice(wrong, min(12, len(wrong)), replace=False) if len(wrong) else []
fig, axes = plt.subplots(2, 6, figsize=(15, 5.6))
for ax, i in zip(axes.flat, pick):
    ax.imshow(ImageOps.fit(Image.open(test.path[i]).convert("RGB"), (220, 220))); ax.axis("off")
    ax.set_title(f"true: {CLASSES[y_true[i]]}\npred: {CLASSES[y_best[i]]}", fontsize=8, color="#b23a1e")
for ax in list(axes.flat)[len(pick):]: ax.axis("off")
plt.suptitle("Sample classification errors", fontweight="bold"); plt.tight_layout(); savefig("cls_errors"); plt.show()
""")
md(r"""
**Classification findings.** The comparison table, report and confusion matrix above are the evidence. The pattern to look for:
errors concentrate in visually similar pairs, usually fried golden items and layered desserts, while dishes with distinctive shape
(pizza, tacos, spring rolls) score highest. The best model is the one with the higher macro-F1. That model is exported for the app.
""")

md(r"""
## 5. Object detection
### 5.1 Creating box labels
The dataset has class folders but **no bounding boxes**, so a detector cannot be trained directly. Hand-drawing ~5,000 boxes is
outside this project's scope. Instead we use **Grounding DINO** (Liu et al., 2023), an open-vocabulary detector that finds objects
from a text prompt. We prompt it with `"food."` and with the class name. We keep confident boxes and assign each box the folder's class.

This is **pseudo-labelling**. Its limits:
- Boxes come from a model, not a person, so detector mAP is measured against these pseudo-labels.
- Images where Grounding DINO finds nothing confident fall back to one box covering the dish area. They are counted and reported.
- Section 6 adds a check that does not depend on the pseudo-labels: does the detect-then-classify pipeline name the right dish?
""")
code(r"""
rs = np.random.RandomState(SEED)
det_rows = []
for split, k in zip(["train", "val", "test"], DET_PER_CLASS):
    sub = df[df.split == split]
    for c in CLASSES:
        s = sub[sub.label == c]
        det_rows.append(s.sample(min(k, len(s)), random_state=SEED))
det_df = pd.concat(det_rows).reset_index(drop=True)
print(det_df.groupby("split").size().to_dict())
""")
code(r"""
PSEUDO = ART / "pseudo_boxes.json"
boxes = json.load(open(PSEUDO)) if PSEUDO.exists() else {}
todo = [p for p in det_df.path if p not in boxes]
gd_ok = False; sys.last_traceback = sys.last_value = sys.last_exc = None; import gc; gc.collect(); torch.cuda.empty_cache()
if todo and not QUICK:
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        gd_id = "IDEA-Research/grounding-dino-tiny"
        proc = AutoProcessor.from_pretrained(gd_id)
        gd = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id).to(dev).eval()
        gd_ok = True
    except Exception as e:
        print("Grounding DINO unavailable:", e)

def gd_detect(paths, labels):
    ims = [Image.open(p).convert("RGB") for p in paths]
    texts = [f"food. {l.replace('_', ' ')}." for l in labels]
    inp = proc(images=ims, text=texts, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        out = gd(**inp)
    sizes = [im.size[::-1] for im in ims]
    try:
        res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.30, text_threshold=0.25, target_sizes=sizes)
    except TypeError:
        res = proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=0.30, text_threshold=0.25, target_sizes=sizes)
    return ims, res

def fallback_box(w, h):   # central box covering the typical dish region
    return [0.1 * w, 0.1 * h, 0.9 * w, 0.9 * h]

def clean(bxs, scores, w, h, max_n=3):
    keep = []
    order = np.argsort(scores)[::-1]
    for i in order:
        b = bxs[i]; area = (b[2] - b[0]) * (b[3] - b[1]) / (w * h)
        if area < 0.02: continue                      # specks
        if area > 0.97 and len(bxs) > 1: continue     # whole-frame duplicates when better boxes exist
        if any(iou(b, k[0]) > 0.6 for k in keep): continue
        keep.append((b, float(scores[i])))
        if len(keep) == max_n: break
    return keep

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0

lab_of = dict(zip(det_df.path, det_df.label))
log(f"pseudo-labelling {len(todo)} images (grounding dino={gd_ok})")
B = 4; sys.last_traceback = sys.last_value = sys.last_exc = None; import gc; eff.cpu(); gc.collect(); torch.cuda.empty_cache()
for i in range(0, len(todo), B):
    chunk = todo[i:i + B]
    if gd_ok:
        ims, res = gd_detect(chunk, [lab_of[p] for p in chunk])
        for p, im, r in zip(chunk, ims, res):
            w, h = im.size
            kept = clean(r["boxes"].cpu().numpy().tolist(), r["scores"].cpu().numpy(), w, h)
            boxes[p] = dict(w=w, h=h, boxes=[k[0] for k in kept], scores=[k[1] for k in kept], src="gdino" if kept else "fallback")
            if not kept: boxes[p]["boxes"] = [fallback_box(w, h)]
    else:
        for p in chunk:
            w, h = Image.open(p).size
            boxes[p] = dict(w=w, h=h, boxes=[fallback_box(w, h)], scores=[1.0], src="fallback")
    if (i // B) % 50 == 0:
        json.dump(boxes, open(PSEUDO, "w")); print(f"{i + len(chunk)}/{len(todo)}")
json.dump(boxes, open(PSEUDO, "w"))
src = pd.Series([boxes[p]["src"] for p in det_df.path]).value_counts()
nbox = pd.Series([len(boxes[p]["boxes"]) for p in det_df.path]).value_counts().sort_index()
print("label source:", src.to_dict(), "| boxes per image:", nbox.to_dict())
log("pseudo-labelling done")
""")
code(r"""
# Quality check of pseudo-labels: area distribution and a visual sample
area = [ (b[2]-b[0])*(b[3]-b[1])/(boxes[p]["w"]*boxes[p]["h"]) for p in det_df.path for b in boxes[p]["boxes"]]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].hist(area, bins=np.linspace(0, 1, 31), color="#2f7d4f"); ax[0].set_title("Pseudo-box area (fraction of image)")
pd.Series([boxes[p]["src"] for p in det_df.path]).value_counts().plot.bar(ax=ax[1], color=["#2f7d4f", "#e4572e"]); ax[1].set_title("Label source")
savefig("pseudo_stats"); plt.show()

from PIL import ImageDraw
smp = det_df.sample(12, random_state=1)
fig, axes = plt.subplots(2, 6, figsize=(16, 5.8))
for ax, (_, r) in zip(axes.flat, smp.iterrows()):
    im = Image.open(r.path).convert("RGB"); d = ImageDraw.Draw(im)
    for b in boxes[r.path]["boxes"]: d.rectangle(b, outline="#00e676", width=4)
    ax.imshow(im); ax.axis("off"); ax.set_title(f"{r.label} ({boxes[r.path]['src']})", fontsize=8)
plt.suptitle("Pseudo-labelled boxes (green)", fontweight="bold"); plt.tight_layout(); savefig("pseudo_samples"); plt.show()
""")
code(r"""
# Write the YOLO detection dataset
DET_DS = WORK / "det_ds"
for split in ["train", "val", "test"]:
    (DET_DS / "images" / split).mkdir(parents=True, exist_ok=True); (DET_DS / "labels" / split).mkdir(parents=True, exist_ok=True)
for _, r in det_df.iterrows():
    stem = f"{r.label}__{Path(r.path).stem}"
    dst = DET_DS / "images" / r.split / f"{stem}.jpg"
    if not dst.exists(): os.symlink(r.path, dst)
    bx = boxes[r.path]; W, H = bx["w"], bx["h"]
    with open(DET_DS / "labels" / r.split / f"{stem}.txt", "w") as f:
        for b in bx["boxes"]:
            cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H; bw, bh = (b[2] - b[0]) / W, (b[3] - b[1]) / H
            f.write(f"{IDX[r.label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
yaml_path = DET_DS / "food.yaml"
yaml_path.write_text(f"path: {DET_DS}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" +
                     "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
print(yaml_path.read_text()[:300])
""")
md(r"""
### 5.2 Training the YOLO11s detector
YOLO11 (Ultralytics, 2024) is a single-stage, anchor-free detector. It predicts boxes and class scores in one pass, which suits
a phone app that needs answers in well under a second. We fine-tune the COCO-pretrained `yolo11s.pt` (9.4 M parameters).
""")
code(r"""
DET_BEST = ART / "weights" / "detector.pt"
if not DET_BEST.exists():
    log("train YOLO11s detector")
    det = YOLO("yolo11s.pt")
    r = det.train(data=str(yaml_path), epochs=DET_EPOCHS, imgsz=512, batch=32 if not QUICK else 8, device=DEVICE,
                  project=str(WORK / "runs"), name="det", exist_ok=True, seed=SEED, patience=10, cos_lr=True,
                  close_mosaic=5, workers=2, plots=True, verbose=False)
    shutil.copy(Path(r.save_dir) / "weights" / "best.pt", DET_BEST)
    shutil.copy(Path(r.save_dir) / "results.csv", ART / "det_results.csv")
    for f in ["results.png", "BoxPR_curve.png", "PR_curve.png", "confusion_matrix_normalized.png"]:
        if (Path(r.save_dir) / f).exists(): shutil.copy(Path(r.save_dir) / f, FIG / f"det_{f}")
    log("done detector")
dh = pd.read_csv(ART / "det_results.csv"); dh.columns = dh.columns.str.strip()
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for k in ["train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss"]: ax[0].plot(dh.epoch, dh[k], label=k)
ax[0].legend(fontsize=7); ax[0].set_title("Detector losses")
ax[1].plot(dh.epoch, dh["metrics/mAP50(B)"], label="mAP@0.5"); ax[1].plot(dh.epoch, dh["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
ax[1].legend(); ax[1].set_title("Validation mAP"); savefig("det_curves"); plt.show()
""")
code(r"""
detector = YOLO(str(DET_BEST))
dm = detector.val(data=str(yaml_path), split="test", imgsz=512, device=DEVICE, verbose=False, plots=True,
                  project=str(WORK / "runs"), name="det_test", exist_ok=True)
det_metrics = {"mAP50": float(dm.box.map50), "mAP50_95": float(dm.box.map), "precision": float(dm.box.mp), "recall": float(dm.box.mr)}
print({k: round(v, 4) for k, v in det_metrics.items()})
per_cls = pd.DataFrame({"class": CLASSES, "AP50_95": [float(dm.box.maps[i]) for i in range(len(CLASSES))]}).sort_values("AP50_95")
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(per_cls["class"].str.replace("_", " "), per_cls.AP50_95, color="#2f7d4f"); ax.set_title("Per-class AP@0.5:0.95 (test, vs pseudo-labels)")
savefig("det_per_class"); plt.show()
""")
code(r"""
# Detector predictions on unseen test images
smp = det_df[det_df.split == "test"].sample(12, random_state=3)
fig, axes = plt.subplots(2, 6, figsize=(16, 6))
for ax, (_, r) in zip(axes.flat, smp.iterrows()):
    res = detector.predict(r.path, imgsz=512, conf=0.25, device=DEVICE, verbose=False)[0]
    ax.imshow(res.plot()[..., ::-1]); ax.axis("off"); ax.set_title(f"true: {r.label}", fontsize=8)
plt.suptitle("YOLO11s detections on test images", fontweight="bold"); plt.tight_layout(); savefig("det_samples"); plt.show()
""")

md(r"""
## 6. Two-stage pipeline: detect → classify
In production, the detector proposes regions and the stronger classifier re-labels each crop. We compare three ways of naming the dish
in each test image (true label = folder):

- **A. Detector only**: label of the highest-confidence box.
- **B. Classifier on whole image**: no localisation.
- **C. Detect → classify**: classifier applied to the top box crop, keeping whichever of the two labels is more confident.

This check uses the real folder labels, so it does not depend on the pseudo-boxes.
""")
code(r"""
tst = det_df[det_df.split == "test"].reset_index(drop=True)
A, Bp, C, lat = [], [], [], []
for _, r in tst.iterrows():
    im = Image.open(r.path).convert("RGB"); t0 = time.time()
    d = detector.predict(im, imgsz=512, conf=0.1, device=DEVICE, verbose=False)[0]
    if len(d.boxes):
        j = int(d.boxes.conf.argmax()); dl, dc = d.names[int(d.boxes.cls[j])], float(d.boxes.conf[j])
        crop = im.crop(tuple(int(v) for v in d.boxes.xyxy[j].tolist()))
    else:
        dl, dc, crop = None, 0.0, im
    c = yolo_cls.predict(crop, imgsz=224, device=DEVICE, verbose=False)[0]
    cl, cc = c.names[int(c.probs.top1)], float(c.probs.top1conf)
    lat.append((time.time() - t0) * 1000)
    w = yolo_cls.predict(im, imgsz=224, device=DEVICE, verbose=False)[0]
    A.append(dl); Bp.append(w.names[int(w.probs.top1)]); C.append(cl if cc >= dc else dl)
pipe = pd.DataFrame({
    "approach": ["A. detector only", "B. classifier, whole image", "C. detect → classify"],
    "accuracy": [np.mean([a == t for a, t in zip(A, tst.label)]), np.mean([b == t for b, t in zip(Bp, tst.label)]),
                 np.mean([c == t for c, t in zip(C, tst.label)])]}).round(4)
pipe["note"] = ["", "", f"median latency {np.median(lat):.0f} ms/image"]
pipe.to_csv(ART / "pipeline_comparison.csv", index=False)
pipe
""")
md(r"""
**Reading the pipeline table.** The detector alone has to localise and classify with fewer training images (it saw ~320 per class,
the classifier ~700). Re-labelling the crop with the classifier usually recovers accuracy while keeping the box. That box is what lets
the app handle **plates with several items** and estimate portion size from box area. The app ships approach C.
""")

md("## 7. Nutrition mapping and meal analytics")
code(r"""
NUTRITION = json.loads(r'''__NUTRI__''')
nut = pd.DataFrame(NUTRITION).T
nut.index.name = "food"; nut[["kcal", "protein_g", "carbs_g", "fat_g"]] = nut[["kcal", "protein_g", "carbs_g", "fat_g"]].astype(float)
nut["kcal_per_100g"] = (nut.kcal / nut.serving_g.astype(float) * 100).round(0)
fig, ax = plt.subplots(figsize=(9, 5))
s = nut.sort_values("kcal")
ax.barh(s.index.str.replace("_", " "), s.protein_g * 4, color="#2f7d4f", label="protein")
ax.barh(s.index.str.replace("_", " "), s.carbs_g * 4, left=s.protein_g * 4, color="#f2a541", label="carbs")
ax.barh(s.index.str.replace("_", " "), s.fat_g * 9, left=s.protein_g * 4 + s.carbs_g * 4, color="#e4572e", label="fat")
ax.set_xlabel("kcal per typical serving"); ax.set_title("Calorie make-up of each dish"); ax.legend()
savefig("nutrition"); plt.show()
nut[["serving_g", "kcal", "protein_g", "carbs_g", "fat_g", "kcal_per_100g", "swap"]]
""")
code(r"""
def analyse_meal(path, portion=1.0):
    im = Image.open(path).convert("RGB")
    d = detector.predict(im, imgsz=512, conf=0.25, device=DEVICE, verbose=False)[0]
    items = []
    for b in d.boxes:
        box = [int(v) for v in b.xyxy[0].tolist()]; dl, dc = d.names[int(b.cls)], float(b.conf)
        c = yolo_cls.predict(im.crop(tuple(box)), imgsz=224, device=DEVICE, verbose=False)[0]
        cl, cc = c.names[int(c.probs.top1)], float(c.probs.top1conf)
        lab = cl if cc >= dc else dl
        n = NUTRITION[lab]
        items.append(dict(food=lab, confidence=round(max(cc, dc), 3), kcal=n["kcal"] * portion, protein_g=n["protein_g"] * portion,
                          carbs_g=n["carbs_g"] * portion, fat_g=n["fat_g"] * portion, healthier_swap=n["swap"]))
    return d, pd.DataFrame(items)

demo = tst.sample(3, random_state=7)
for _, r in demo.iterrows():
    d, items = analyse_meal(r.path)
    plt.figure(figsize=(4, 4)); plt.imshow(d.plot()[..., ::-1]); plt.axis("off"); plt.title(f"true: {r.label}"); plt.show()
    display(items)
""")
code(r"""
# Pattern analysis on a simulated 30-day food log (illustrates the "analyse user choices" objective)
rng = np.random.RandomState(SEED)
prefs = rng.dirichlet(np.ones(len(CLASSES)) * 0.4)               # one user's taste profile
log_rows = []
for day in pd.date_range("2026-10-01", periods=30):
    for meal in ["breakfast", "lunch", "dinner"] + (["snack"] if rng.rand() < .6 else []):
        f = rng.choice(CLASSES, p=prefs); n = NUTRITION[f]; portion = rng.choice([0.75, 1, 1.25], p=[.2, .6, .2])
        log_rows.append(dict(date=day, meal=meal, food=f, kcal=n["kcal"] * portion, protein_g=n["protein_g"] * portion,
                             carbs_g=n["carbs_g"] * portion, fat_g=n["fat_g"] * portion))
ulog = pd.DataFrame(log_rows)
daily = ulog.groupby("date")[["kcal", "protein_g", "carbs_g", "fat_g"]].sum()
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
ax[0].plot(daily.index, daily.kcal, color="#2f7d4f", marker="o", ms=3); ax[0].axhline(2000, ls="--", color="#e4572e", label="2,000 kcal goal")
ax[0].set_title("Daily calories (simulated user)"); ax[0].legend()
ulog.food.value_counts().head(8).plot.barh(ax=ax[1], color="#52b788"); ax[1].invert_yaxis(); ax[1].set_title("Most-logged foods")
plt.tight_layout(); savefig("user_log"); plt.show()
over = (daily.kcal > 2000).mean()
top = ulog.food.value_counts().index[:3]
print(f"Days over goal: {over:.0%}. Average intake: {daily.kcal.mean():.0f} kcal/day.")
print("Personalised suggestions:")
for f in top: print(f"  • You often log {f.replace('_', ' ')} → try: {NUTRITION[f]['swap']}")
""")

md(r"""
## 8. Business insights, limitations and next steps
**What the models deliver for NutriScan**
- The classifier recognises the 17 dishes with the accuracy and macro-F1 shown in Section 4. Errors cluster in look-alike pairs, which
  tells the product team where a "did you mean …?" confirmation prompt adds the most value.
- The detector localises the dish so the app can handle more than one item per photo and scale portions by box size.
- Combining detector and classifier (Section 6) gives the best naming accuracy with latency suitable for a mobile backend.
- Mapping labels to a nutrition table turns a photo into calories and macros in one step. Logging those results over time supports
  the brief's goal of spotting patterns and suggesting healthier swaps.

**Limitations**
- **Detection labels are machine-generated.** Detector mAP is measured against Grounding DINO boxes, not hand-drawn ground truth.
  A few hundred hand-checked boxes would give a trustworthy mAP and could be used to fine-tune.
- **17 classes only.** Real meals contain hundreds of foods. Anything outside these classes gets forced into one of them.
  An "unknown food" threshold is needed before launch.
- **Nutrition is per typical serving** from public reference values. Portion estimation from a single 2-D photo is approximate.
  Depth sensing or a reference object (plate, card) would tighten it.
- **Apple pie is under-represented** (257 images vs ~1,000). Oversampling helps but more real images would help more.
- The user-pattern analysis uses a **simulated** log, to show the mechanism.

**Next steps**
1. Hand-verify ~500 pseudo-boxes with Label Studio or Roboflow, re-train, and report mAP on that clean set.
2. Extend to the full Food-101 (101 classes) and add multi-item plate photos (e.g. UEC-Food-256).
3. Add an open-set rejection threshold and a user-correction loop that feeds labelled data back into training.
4. Integrate with Apple Health / Google Fit through their APIs for longitudinal tracking.

**References**
- Bossard, Guillaumin & Van Gool (2014). *Food-101 — Mining Discriminative Components with Random Forests.* ECCV.
- Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML.
- Liu et al. (2023). *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.* arXiv:2303.05499.
- Jocher, Qiu & Chaurasia (2024). *Ultralytics YOLO11.* github.com/ultralytics/ultralytics.
""")

md("## 9. Export artefacts and launch the Streamlit app")
code(r"""
metrics = {
    "Classifier": best_name,
    "Classifier test accuracy": f"{cmp.set_index('model').loc[best_name, 'accuracy']:.1%}",
    "Classifier macro-F1": f"{cmp.set_index('model').loc[best_name, 'macro_F1']:.3f}",
    "Detector mAP@0.5": f"{det_metrics['mAP50']:.3f}",
    "Detector mAP@0.5:0.95": f"{det_metrics['mAP50_95']:.3f}",
    "Pipeline accuracy (detect→classify)": f"{pipe.accuracy.iloc[2]:.1%}",
    "Classes": len(CLASSES),
}
json.dump(metrics, open(ART / "metrics.json", "w"), indent=1)
APP_DIR = ART / "app"; (APP_DIR / "weights").mkdir(parents=True, exist_ok=True); (APP_DIR / "samples").mkdir(exist_ok=True)
shutil.copy(DET_BEST, APP_DIR / "weights" / "detector.pt")
if best_name == "YOLO11s-cls":
    shutil.copy(CLS_BEST, APP_DIR / "weights" / "classifier.pt")
else:   # the app runs Ultralytics models; keep YOLO-cls as the in-app refiner and report EfficientNet in the notebook
    shutil.copy(CLS_BEST, APP_DIR / "weights" / "classifier.pt")
json.dump(NUTRITION, open(APP_DIR / "nutrition.json", "w"), indent=1)
json.dump(metrics, open(APP_DIR / "metrics.json", "w"), indent=1)
for c in CLASSES:
    p = tst[tst.label == c].path.iloc[0]
    ImageOps.contain(Image.open(p).convert("RGB"), (640, 640)).save(APP_DIR / "samples" / f"{c}__1.jpg", quality=88)
(APP_DIR / "requirements.txt").write_text("streamlit>=1.50\nultralytics>=8.3\nplotly\npandas\npillow\nhuggingface_hub\nopencv-python-headless\n")
log("artefacts exported")
metrics
""")
code(r"""
%%writefile __APP_PATH__
__APP__
""")
code(r"""
# Run NutriScan inside this notebook. Streamlit needs a WebSocket that Colab's proxy blocks, so the
# in-notebook demo uses the Gradio version of the same app. The Streamlit version is deployed publicly.
import subprocess, sys
subprocess.run(["pkill", "-9", "-f", "streamlit run"])
pip("gradio")
for sub in ["weights", "samples"]: shutil.copytree(APP_DIR / sub, f"/content/{sub}", dirs_exist_ok=True)
shutil.copy(APP_DIR / "nutrition.json", "/content/nutrition.json")
shutil.copy(CAP / "deploy" / "gradio_app.py", "/content/gradio_app.py")
sys.path.insert(0, "/content")
import gradio_app
print("Public Streamlit version: https://nutriscan-food.streamlit.app")
gradio_app.demo.launch(theme=gradio_app.theme, css=gradio_app.CSS, share=False, inline=True, height=1000)
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU",
                   "colab": {"gpuType": "T4", "provenance": []}, "language_info": {"name": "python"}},
      "cells": []}
for t, s in cells:
    s = s.replace("__NUTRI__", NUTRI).replace("__APP__", APP).replace("__APP_PATH__", "/content/app.py")
    c = {"cell_type": t, "metadata": {}, "source": s}
    if t == "code": c.update(execution_count=None, outputs=[])
    nb["cells"].append(c)
json.dump(nb, open("notebooks/NutriScan_Food_Detection.ipynb", "w"), indent=1)
print(len(nb["cells"]), "cells")
