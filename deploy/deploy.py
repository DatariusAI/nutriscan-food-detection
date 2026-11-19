# Publishes NutriScan to the Hugging Face Hub. Run inside the Colab kernel after training:
#   from google.colab import userdata; import os; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
#   %run deploy.py
import glob, os, shutil, json
from pathlib import Path
from huggingface_hub import HfApi

CAP = Path(glob.glob("/content/drive/MyDrive/Colab Notebooks/Training_Great Learning/MDS*/Capstone")[0])
ART, SRC = CAP / "nutriscan_artifacts", CAP / "deploy"
USER = "DatariusAI"
MODEL_REPO, SPACE_REPO = f"{USER}/nutriscan-food-models", f"{USER}/nutriscan"
api = HfApi(token=os.environ["HF_TOKEN"])
metrics = json.load(open(ART / "metrics.json"))

# 1) Model repo: weights + figures + model card
api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=False)
card = ["---", "license: mit", "library_name: ultralytics", "tags: [food, object-detection, image-classification, yolo11]", "---",
        "# NutriScan food models", "", "YOLO11s detector + YOLO11s-cls classifier for 17 Food-101 classes.", "",
        "| Metric | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in metrics.items()] + [
        "", "Code and notebook: https://github.com/DatariusAI/nutriscan-food-detection",
        "", "Detection boxes were auto-generated with Grounding DINO (pseudo-labels); see the notebook for details."]
stage = Path("/content/hf_model"); shutil.rmtree(stage, ignore_errors=True); (stage / "weights").mkdir(parents=True)
for f in ["detector.pt", "classifier.pt"]: shutil.copy(ART / "weights" / f, stage / "weights" / f)
if (ART / "weights" / "efficientnet_b0.pt").exists(): shutil.copy(ART / "weights" / "efficientnet_b0.pt", stage / "weights")
shutil.copytree(ART / "figures", stage / "figures")
shutil.copytree(ART / "app" / "samples", stage / "samples")
for nb in CAP.glob("NutriScan_Food_Detection.ipynb"): shutil.copy(nb, stage / nb.name)
for f in ["metrics.json", "classifier_comparison.csv", "pipeline_comparison.csv"]:
    if (ART / f).exists(): shutil.copy(ART / f, stage / f)
(stage / "README.md").write_text("\n".join(card))
api.upload_folder(folder_path=str(stage), repo_id=MODEL_REPO, repo_type="model", commit_message="NutriScan models + figures")
print("model repo:", f"https://huggingface.co/{MODEL_REPO}")

# 2) Space: docker Streamlit app with bundled weights
space = Path("/content/hf_space"); shutil.rmtree(space, ignore_errors=True)
shutil.copytree(ART / "app", space)
for f in ["app.py", "Dockerfile", "README.md", "requirements.txt"]: shutil.copy(SRC / f, space / f)
(space / ".streamlit").mkdir(exist_ok=True); shutil.copy(SRC / "config.toml", space / ".streamlit" / "config.toml")
api.create_repo(SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
api.upload_folder(folder_path=str(space), repo_id=SPACE_REPO, repo_type="space", commit_message="Deploy NutriScan app")
print("space:", f"https://huggingface.co/spaces/{SPACE_REPO}")
