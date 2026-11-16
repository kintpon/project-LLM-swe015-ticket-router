"""Upload the trained models/ folder to the public Hugging Face repo named in config.py.
Run `huggingface-cli login` yourself first; this script never reads or stores a token.
Only the files the app needs are uploaded (no smoke models, no cached teacher logits, no fp32 ONNX)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `python scripts/upload_weights.py` finds config.py

from huggingface_hub import HfApi  # noqa: E402

from config import HF_MODEL_REPO, MODELS_DIR  # noqa: E402

SKIP = ["smoke/*", "m3_train_logits.npy", "m3s_fp32.onnx"]

if __name__ == "__main__":
    if not (MODELS_DIR / "m4_int8.onnx").exists():
        raise SystemExit("models/m4_int8.onnx not found - run python train_all.py first")
    api = HfApi()
    api.create_repo(HF_MODEL_REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(MODELS_DIR), repo_id=HF_MODEL_REPO, ignore_patterns=SKIP,
                      commit_message="Upload trained ticket-router models")
    print(f"uploaded models/ to https://huggingface.co/{HF_MODEL_REPO}")
