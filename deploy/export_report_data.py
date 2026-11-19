# Run inside the live Colab kernel after the notebook finishes:  exec(open(str(CAP/'export_report_data.py')).read())
import json as _j
from PIL import Image as _I
R = ART / "report"; R.mkdir(exist_ok=True)
def _f(x):
    try: return float(x)
    except Exception: return x
out = {
    "class_counts": df.label.value_counts().to_dict(),
    "split_counts": df.split.value_counts().to_dict(),
    "n_images": int(len(df)), "n_corrupt": int(len(bad)), "n_dups": int(len(dups)),
    "img_stats": df[["w", "h", "aspect", "bright"]].describe().round(2).to_dict(),
    "cls_hist": pd.read_csv(ART / "cls_results.csv").rename(columns=lambda c: c.strip()).to_dict("list"),
    "eff_hist": eff_hist,
    "cls_compare": cmp.to_dict("records"),
    "best_classifier": best_name,
    "class_report": {k: ({kk: _f(vv) for kk, vv in v.items()} if isinstance(v, dict) else _f(v)) for k, v in rep.items()},
    "confusion": np.round(cm, 4).tolist(), "classes": CLASSES,
    "top_confusions": top_conf.to_dict("records"),
    "pseudo_src": pd.Series([boxes[p]["src"] for p in det_df.path]).value_counts().to_dict(),
    "pseudo_nbox": {int(k): int(v) for k, v in pd.Series([len(boxes[p]["boxes"]) for p in det_df.path]).value_counts().sort_index().items()},
    "pseudo_area": np.histogram(area, bins=np.linspace(0, 1, 21))[0].tolist(),
    "det_split": det_df.split.value_counts().to_dict(),
    "det_hist": pd.read_csv(ART / "det_results.csv").rename(columns=lambda c: c.strip()).to_dict("list"),
    "det_metrics": det_metrics,
    "det_per_class": per_cls.to_dict("records"),
    "pipeline": pipe.to_dict("records"),
    "pipeline_latency_ms": {"median": float(np.median(lat)), "p90": float(np.percentile(lat, 90))},
    "metrics": metrics,
    "user_log": {"days_over_goal": float(over), "avg_kcal": float(daily.kcal.mean()), "top_foods": ulog.food.value_counts().head(5).to_dict()},
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "epochs": {"cls": CLS_EPOCHS, "eff": EFF_EPOCHS, "det": DET_EPOCHS, "det_per_class": DET_PER_CLASS},
    "colab_app_url": (ART / "colab_app_url.txt").read_text() if (ART / "colab_app_url.txt").exists() else None,
}
_j.dump(out, open(R / "report_data.json", "w"), default=str, indent=1)
for name in ["pseudo_samples", "det_samples", "cls_errors"]:
    p = FIG / f"{name}.png"
    if p.exists():
        im = _I.open(p).convert("RGB"); im.thumbnail((1100, 1100)); im.save(R / f"{name}.jpg", quality=55, optimize=True)
print("report data exported:", sorted(x.name for x in R.iterdir()), [(x.name, x.stat().st_size) for x in R.iterdir()])
