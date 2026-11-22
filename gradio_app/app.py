"""NutriScan - Gradio edition. Detect food (YOLO11s), name it (YOLO11s-cls), and turn it into nutrition."""
import json, os, time
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).parent
REPO = os.environ.get("NUTRISCAN_MODEL_REPO", "DatariusAI/nutriscan-food-models")


def local_or_hub(rel):
    p = ROOT / rel
    if p.exists():
        return str(p)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, rel)


DET = YOLO(local_or_hub("weights/detector.pt"))
CLS = YOLO(local_or_hub("weights/classifier.pt"))
NUTRI = json.load(open(local_or_hub("nutrition.json")))
PALETTE = ["#2f7d4f", "#e4572e", "#f2a541", "#3a86ff", "#8338ec", "#ff006e", "#06a77d", "#ffbe0b"]
pretty = lambda s: s.replace("_", " ").title()


def analyse(img, portion, conf, refine):
    if img is None:
        return None, "<p class='hint'>Upload or snap a meal photo to begin.</p>", None, None
    img = img.convert("RGB"); t0 = time.time(); items = []
    r = DET.predict(img, conf=conf, iou=0.5, imgsz=640, verbose=False)[0]
    for b in r.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        lab, sc = r.names[int(b.cls)], float(b.conf)
        crop = img.crop((x1, y1, x2, y2))
        if refine and crop.width > 8 and crop.height > 8:
            c = CLS.predict(crop, imgsz=224, verbose=False)[0]
            if float(c.probs.top1conf) >= sc:
                lab, sc = c.names[int(c.probs.top1)], float(c.probs.top1conf)
        items.append(dict(label=lab, conf=sc, box=(x1, y1, x2, y2)))
    ms = (time.time() - t0) * 1000
    if not items:  # no known dish detected: say so rather than force the photo into the nearest class
        known = ", ".join(pretty(k) for k in sorted(NUTRI))
        return img, (f"<div class='swaps'><h4>No known dish found</h4><p>I couldn't find any of the 17 dishes I know, so I won't guess.</p>"
                     f"<p><b>Supported:</b> {known}.</p></div>"), None, pd.DataFrame()

    out = img.copy(); d = ImageDraw.Draw(out); lw = max(3, img.width // 220)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, img.width // 45))
    except Exception:
        font = ImageFont.load_default()
    rows = []
    for i, it in enumerate(items):
        col = PALETTE[i % len(PALETTE)]; x1, y1, x2, y2 = it["box"]
        d.rounded_rectangle((x1, y1, x2, y2), radius=lw * 3, outline=col, width=lw)
        txt = f" {pretty(it['label'])} {it['conf']:.0%} "; tb = d.textbbox((x1, y1), txt, font=font); h = tb[3] - tb[1] + 8
        ty = max(0, y1 - h); d.rounded_rectangle((x1, ty, x1 + tb[2] - tb[0] + 6, ty + h), radius=6, fill=col)
        d.text((x1 + 3, ty + 2), txt, fill="white", font=font)
        n = NUTRI[it["label"]]
        rows.append(dict(Food=pretty(it["label"]), Confidence=f"{it['conf']:.0%}", kcal=round(n["kcal"] * portion),
                         Protein_g=round(n["protein_g"] * portion, 1), Carbs_g=round(n["carbs_g"] * portion, 1),
                         Fat_g=round(n["fat_g"] * portion, 1), Swap=n["swap"]))
    df = pd.DataFrame(rows)
    k, p, c, f = df.kcal.sum(), df.Protein_g.sum(), df.Carbs_g.sum(), df.Fat_g.sum()
    cards = "".join(f"<div class='kpi'><b>{v}</b><span>{l}</span></div>" for v, l in
                    [(f"{k:.0f}", "kcal"), (f"{p:.0f} g", "protein"), (f"{c:.0f} g", "carbs"), (f"{f:.0f} g", "fat")])
    swaps = "".join(f"<li><b>{r.Food}</b> → {r.Swap}</li>" for r in df.itertuples())
    html = f"<div class='kpis'>{cards}</div><p class='meta'>{len(df)} item(s) · {ms:.0f} ms</p><div class='swaps'><h4>💡 Healthier swaps</h4><ul>{swaps}</ul></div>"
    fig = go.Figure(go.Pie(labels=["Protein", "Carbs", "Fat"], values=[p * 4, c * 4, f * 9], hole=.66, sort=False,
                           marker=dict(colors=["#2f7d4f", "#f2a541", "#e4572e"])))
    fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)",
                      annotations=[dict(text=f"<b>{k:.0f}</b><br>kcal", showarrow=False, font_size=18)], legend=dict(orientation="h"))
    return out, html, fig, df


CSS = """
.gradio-container{max-width:1200px!important;background:radial-gradient(900px 400px at 0% -10%,#e8f5ec 0%,transparent 60%),#fbf8f1}
#hero{background:linear-gradient(120deg,#16432c,#2f7d4f 55%,#52b788);color:#fff;border-radius:22px;padding:26px 30px;margin-bottom:8px}
#hero h1{color:#fff;font-size:2.3rem;margin:0;font-family:Georgia,serif} #hero p{color:#e8f5ec;margin:.4rem 0 0}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.kpi{background:#fff;border:1px solid #e9e4d8;border-radius:16px;padding:12px;text-align:center}
.kpi b{display:block;font-size:1.6rem;color:#1d2a22;font-family:Georgia,serif} .kpi span{color:#5d6b62;font-size:.8rem;text-transform:uppercase}
.meta{color:#5d6b62;font-size:.85rem} .swaps{background:#fdeee6;border-radius:14px;padding:10px 16px;margin-top:8px}
.swaps h4{margin:0 0 4px;color:#8a3b1f} .swaps li{color:#5a2a14} .hint{color:#5d6b62}
"""
theme = gr.themes.Soft(primary_hue="green", secondary_hue="orange", neutral_hue="stone",
                       font=[gr.themes.GoogleFont("Inter"), "sans-serif"])
samples_dir = ROOT / "samples"
examples = [[str(p)] for p in sorted(samples_dir.glob("*.jpg"))][:8] if samples_dir.exists() else None

with gr.Blocks(title="NutriScan · AI Food Detection") as demo:
    gr.HTML("<div id='hero'><h1>🥗 NutriScan</h1><p>Snap a meal. NutriScan boxes every food item, names it, and turns it into calories, macros and a healthier swap.</p></div>")
    with gr.Row():
        with gr.Column(scale=5):
            inp = gr.Image(type="pil", label="Meal photo", sources=["upload", "webcam", "clipboard"], height=380)
            with gr.Row():
                portion = gr.Slider(0.5, 2.0, 1.0, step=0.25, label="Portion (× serving)")
                conf = gr.Slider(0.05, 0.9, 0.45, step=0.05, label="Detection confidence")
            refine = gr.Checkbox(True, label="Refine labels with the classifier")
            btn = gr.Button("Scan my plate", variant="primary", size="lg")
            if examples:
                gr.Examples(examples, inputs=inp, label="Try a sample")
        with gr.Column(scale=6):
            out_img = gr.Image(label="Detected items", height=380)
            summary = gr.HTML()
            donut = gr.Plot(label="Macro split")
    table = gr.Dataframe(label="Per-item nutrition", wrap=True)
    gr.Markdown("Nutrition values are typical-serving estimates for guidance only, not medical advice. · "
                "[Code & report](https://github.com/DatariusAI/nutriscan-food-detection) · "
                "[Streamlit version](https://nutriscan-food.streamlit.app)")
    btn.click(analyse, [inp, portion, conf, refine], [out_img, summary, donut, table])
    inp.change(analyse, [inp, portion, conf, refine], [out_img, summary, donut, table])

if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS)
