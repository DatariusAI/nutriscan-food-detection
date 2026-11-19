"""NutriScan - AI food detection & nutrition analysis (Streamlit).

Pipeline: YOLO detector finds food regions -> YOLO classifier labels each crop
(falls back to the detector's own label) -> nutrition lookup -> meal summary.
"""
import io, json, os, time
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
DET_W = ROOT / "weights" / "detector.pt"
CLS_W = ROOT / "weights" / "classifier.pt"
NUTRI = json.load(open(ROOT / "nutrition.json"))
HUB_REPO = os.environ.get("NUTRISCAN_MODEL_REPO", "DatariusAI/nutriscan-food-models")


def hub_file(name):
    """Local file if bundled, otherwise fetched from the public Hugging Face model repo."""
    local = ROOT / name
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(HUB_REPO, name))
    except Exception:
        return None


_m = hub_file("metrics.json")
METRICS = json.load(open(_m)) if _m else {}

st.set_page_config(page_title="NutriScan · AI Food Detection", page_icon="🥗", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=Inter:wght@400;500;600&display=swap');
:root{--ink:#1d2a22;--muted:#5d6b62;--leaf:#2f7d4f;--leaf2:#52b788;--cream:#fbf8f1;--tomato:#e4572e;--card:#ffffff;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:radial-gradient(1200px 500px at 10% -10%,#e8f5ec 0%,transparent 60%),
       radial-gradient(900px 400px at 110% 0%,#fdeee6 0%,transparent 55%),var(--cream);}
.block-container{padding-top:1.4rem;max-width:1250px;}
h1,h2,h3{font-family:'Fraunces',serif!important;color:var(--ink);letter-spacing:-.01em;}
.hero{display:flex;justify-content:space-between;align-items:flex-end;gap:2rem;padding:1.6rem 2rem;border-radius:22px;
      background:linear-gradient(120deg,#16432c 0%,#2f7d4f 55%,#52b788 100%);color:#fff;box-shadow:0 18px 40px -18px #16432c88;margin-bottom:1.2rem;}
.hero h1{color:#fff!important;font-size:2.6rem;margin:0;line-height:1.05;}
.hero p{margin:.4rem 0 0;opacity:.88;font-size:1.02rem;max-width:560px;}
.pill{display:inline-block;padding:.25rem .7rem;border-radius:999px;background:#ffffff22;border:1px solid #ffffff44;font-size:.78rem;margin-right:.35rem;}
.card{background:var(--card);border-radius:18px;padding:1.1rem 1.2rem;border:1px solid #e9e4d8;box-shadow:0 8px 24px -16px #1d2a2240;}
.kpi{font-family:'Fraunces',serif;font-size:2.1rem;font-weight:800;color:var(--ink);line-height:1;}
.kpi-l{color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;margin-top:.35rem;}
.item{display:flex;gap:1rem;align-items:center;background:#fff;border:1px solid #e9e4d8;border-radius:16px;padding:.75rem;margin-bottom:.7rem;}
.item img{width:84px;height:84px;object-fit:cover;border-radius:12px;}
.item .nm{font-family:'Fraunces',serif;font-weight:600;font-size:1.12rem;color:var(--ink);}
.tag{font-size:.72rem;padding:.12rem .5rem;border-radius:999px;background:#e8f5ec;color:var(--leaf);margin-left:.4rem;}
.macro{font-size:.85rem;color:var(--muted);}
.swap{font-size:.85rem;color:#8a3b1f;background:#fdeee6;border-radius:10px;padding:.35rem .6rem;margin-top:.35rem;display:inline-block;}
.conf{height:6px;border-radius:6px;background:#eee;overflow:hidden;margin-top:.4rem;width:180px;}
.conf>div{height:100%;background:linear-gradient(90deg,var(--leaf2),var(--leaf));}
section[data-testid="stSidebar"]{background:#16432c;}
section[data-testid="stSidebar"] *{color:#eaf4ee!important;}
.stButton>button,.stDownloadButton>button{border-radius:12px;border:0;background:var(--leaf);color:#fff;font-weight:600;}
.stTabs [data-baseweb="tab"]{font-weight:600;}
footer{visibility:hidden;}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    from ultralytics import YOLO
    repo = os.environ.get("NUTRISCAN_MODEL_REPO", "DatariusAI/nutriscan-food-models")
    paths = {}
    for key, local in (("det", DET_W), ("cls", CLS_W)):
        if local.exists():
            paths[key] = str(local)
        else:  # e.g. Streamlit Cloud: pull weights from the Hugging Face model repo
            try:
                from huggingface_hub import hf_hub_download
                paths[key] = hf_hub_download(repo, f"weights/{local.name}")
            except Exception:
                paths[key] = None
    det = YOLO(paths["det"]) if paths["det"] else None
    cls = YOLO(paths["cls"]) if paths["cls"] else None
    return det, cls


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


PALETTE = ["#2f7d4f", "#e4572e", "#f2a541", "#3a86ff", "#8338ec", "#ff006e", "#06a77d", "#ffbe0b"]


def analyse(img: Image.Image, conf: float, use_cls: bool, portion: float):
    det, cls = load_models()
    rgb = img.convert("RGB")
    t0 = time.time()
    items = []
    if det is not None:
        r = det.predict(rgb, conf=conf, iou=0.5, verbose=False, imgsz=640)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            label, score = r.names[int(b.cls)], float(b.conf)
            crop = rgb.crop((x1, y1, x2, y2))
            src = "detector"
            if use_cls and cls is not None and crop.width > 8 and crop.height > 8:
                c = cls.predict(crop, verbose=False, imgsz=224)[0]
                cl, cs = c.names[int(c.probs.top1)], float(c.probs.top1conf)
                if cs >= score:
                    label, score, src = cl, cs, "classifier"
            items.append(dict(label=label, conf=score, box=(x1, y1, x2, y2), crop=crop, src=src))
    if not items and cls is not None:  # no box found -> whole-image classification
        c = cls.predict(rgb, verbose=False, imgsz=224)[0]
        items.append(dict(label=c.names[int(c.probs.top1)], conf=float(c.probs.top1conf),
                          box=(0, 0, rgb.width, rgb.height), crop=rgb, src="classifier (whole image)"))
    ms = (time.time() - t0) * 1000

    # draw
    out = rgb.copy()
    d = ImageDraw.Draw(out)
    lw = max(3, rgb.width // 220)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, rgb.width // 45))
    except Exception:
        font = ImageFont.load_default()
    for i, it in enumerate(items):
        col = PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = it["box"]
        d.rounded_rectangle((x1, y1, x2, y2), radius=lw * 3, outline=col, width=lw)
        txt = f" {pretty(it['label'])} {it['conf']:.0%} "
        tb = d.textbbox((x1, y1), txt, font=font)
        h = tb[3] - tb[1] + 8
        ty = max(0, y1 - h)
        d.rounded_rectangle((x1, ty, x1 + tb[2] - tb[0] + 6, ty + h), radius=6, fill=col)
        d.text((x1 + 3, ty + 2), txt, fill="white", font=font)
        n = NUTRI.get(it["label"], dict(serving_g=150, kcal=300, protein_g=10, carbs_g=35, fat_g=12,
                                        group="Food", swap="—"))
        it.update({k: (v * portion if isinstance(v, (int, float)) else v) for k, v in n.items()})
    return out, items, ms


def donut(p, c, f):
    kc = [p * 4, c * 4, f * 9]
    fig = go.Figure(go.Pie(labels=["Protein", "Carbs", "Fat"], values=kc, hole=.68, sort=False,
                           marker=dict(colors=["#2f7d4f", "#f2a541", "#e4572e"]), textinfo="percent"))
    fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), showlegend=True,
                      legend=dict(orientation="h", y=-.08), paper_bgcolor="rgba(0,0,0,0)",
                      annotations=[dict(text=f"<b>{sum(kc):.0f}</b><br>kcal", showarrow=False, font_size=18)])
    return fig


def gauge(val, goal):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=val, number=dict(suffix=" kcal"),
        gauge=dict(axis=dict(range=[0, goal]), bar=dict(color="#2f7d4f"),
                   steps=[dict(range=[0, goal * .8], color="#e8f5ec"), dict(range=[goal * .8, goal], color="#fdeee6")])))
    fig.update_layout(height=230, margin=dict(t=30, b=0, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def img_b64(im: Image.Image) -> str:
    import base64
    im = im.copy(); im.thumbnail((200, 200))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🥗 NutriScan")
    st.caption("Snap a meal. See what's on the plate.")
    goal = st.number_input("Daily calorie goal", 1200, 4000, 2000, 50)
    portion = st.select_slider("Portion size", options=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0], value=1.0,
                               format_func=lambda x: f"{x}× serving")
    conf = st.slider("Detection confidence", 0.05, 0.9, 0.25, 0.05)
    use_cls = st.toggle("Refine labels with classifier", value=True)
    st.divider()
    det, cls = load_models()
    st.markdown(f"**Detector:** {'✅ loaded' if det else '⚠️ missing'}  \n**Classifier:** {'✅ loaded' if cls else '⚠️ missing'}")
    if METRICS:
        st.markdown("**Model scorecard**")
        for k, v in METRICS.items():
            st.markdown(f"- {k}: **{v}**")

if "log" not in st.session_state:
    st.session_state.log = []

# ----------------------------------------------------------------- hero
st.markdown(f"""<div class="hero"><div>
<span class="pill">YOLO detection</span><span class="pill">Food classification</span><span class="pill">{len(NUTRI)} dishes</span>
<h1>Know your plate<br>in one photo.</h1>
<p>NutriScan finds every food item in your picture, names it, and turns it into calories and macros — with a healthier swap for each.</p>
</div><div style="font-size:4.2rem;line-height:1">🍱</div></div>""", unsafe_allow_html=True)

tab_scan, tab_log, tab_about = st.tabs(["📸  Scan a meal", "📒  Today's log", "ℹ️  How it works"])

with tab_scan:
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        src = st.radio("Image source", ["Upload", "Camera", "Sample"], horizontal=True, label_visibility="collapsed")
        img = None
        if src == "Upload":
            f = st.file_uploader("Drop a food photo", type=["jpg", "jpeg", "png", "webp"])
            if f: img = Image.open(f)
        elif src == "Camera":
            f = st.camera_input("Take a photo")
            if f: img = Image.open(f)
        else:
            samples = sorted((ROOT / "samples").glob("*.jpg")) if (ROOT / "samples").exists() else []
            if not samples:
                got = [hub_file(f"samples/{c}__1.jpg") for c in NUTRI]
                samples = [p for p in got if p]
            if samples:
                pick = st.selectbox("Sample image", samples, format_func=lambda p: pretty(p.stem.split("__")[0]))
                img = Image.open(pick)
            else:
                st.info("No sample images bundled.")
    if img is not None:
        with st.spinner("Scanning your plate…"):
            annotated, items, ms = analyse(img, conf, use_cls, portion)
        with c1:
            st.image(annotated, width="stretch")
            st.caption(f"{len(items)} item(s) · {ms:.0f} ms")
        with c2:
            if not items:
                st.warning("No food found. Try lowering the confidence threshold.")
            else:
                tk = sum(i["kcal"] for i in items); tp = sum(i["protein_g"] for i in items)
                tc = sum(i["carbs_g"] for i in items); tf = sum(i["fat_g"] for i in items)
                k1, k2, k3, k4 = st.columns(4)
                for col, v, l in [(k1, f"{tk:.0f}", "kcal"), (k2, f"{tp:.0f}g", "protein"),
                                  (k3, f"{tc:.0f}g", "carbs"), (k4, f"{tf:.0f}g", "fat")]:
                    col.markdown(f'<div class="card"><div class="kpi">{v}</div><div class="kpi-l">{l}</div></div>',
                                 unsafe_allow_html=True)
                st.plotly_chart(donut(tp, tc, tf), width="stretch", config={"displayModeBar": False})
                for it in items:
                    st.markdown(f"""<div class="item"><img src="data:image/jpeg;base64,{img_b64(it['crop'])}"/>
<div><div class="nm">{pretty(it['label'])}<span class="tag">{it['group']}</span></div>
<div class="macro">{it['kcal']:.0f} kcal · P {it['protein_g']:.0f}g · C {it['carbs_g']:.0f}g · F {it['fat_g']:.0f}g · ~{it['serving_g']:.0f} g</div>
<div class="conf"><div style="width:{it['conf']*100:.0f}%"></div></div>
<div class="swap">💡 Swap: {it['swap']}</div></div></div>""", unsafe_allow_html=True)
                if st.button("➕ Add this meal to today's log", width="stretch"):
                    for it in items:
                        st.session_state.log.append(dict(time=time.strftime("%H:%M"), food=pretty(it["label"]),
                            kcal=round(it["kcal"]), protein_g=round(it["protein_g"]),
                            carbs_g=round(it["carbs_g"]), fat_g=round(it["fat_g"])))
                    st.success("Logged!")
    else:
        with c2:
            st.markdown('<div class="card"><h3>How to use</h3><p>1. Upload or snap a meal photo.<br>'
                        '2. NutriScan boxes each food item and names it.<br>3. Review calories, macros and swaps.<br>'
                        '4. Add the meal to your daily log.</p></div>', unsafe_allow_html=True)

with tab_log:
    log = pd.DataFrame(st.session_state.log)
    if log.empty:
        st.info("Nothing logged yet. Scan a meal and press **Add to today's log**.")
    else:
        a, b = st.columns([1, 1.3], gap="large")
        with a:
            st.plotly_chart(gauge(log.kcal.sum(), goal), width="stretch", config={"displayModeBar": False})
            left = goal - log.kcal.sum()
            st.markdown(f"**{abs(left):.0f} kcal {'left' if left >= 0 else 'over'}** your goal of {goal}.")
        with b:
            st.dataframe(log, width="stretch", hide_index=True)
            st.download_button("⬇️ Export log (CSV)", log.to_csv(index=False), "nutriscan_log.csv", "text/csv")
        if st.button("Clear log"):
            st.session_state.log = []; st.rerun()

with tab_about:
    st.markdown("""
### Two-stage pipeline
1. **Detection** — a YOLO model fine-tuned on the food images finds each item and draws a box around it.
2. **Classification** — each box is cropped and passed to a YOLO classification model fine-tuned on the same classes.
   The app keeps whichever label has higher confidence.
3. **Nutrition** — the label is looked up in a per-serving nutrition table and scaled by the portion slider.

Nutrition values are approximate per typical serving and are for guidance only, not medical advice.
""")
    if METRICS:
        st.json(METRICS)
