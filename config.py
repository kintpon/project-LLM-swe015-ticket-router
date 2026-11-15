"""Central settings: paths, seed, and the Hugging Face / GitHub names."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42

DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
APPLIANCE_CSV = DATA_DIR / "appliance.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

HF_MODEL_REPO = "kintpon/swe015-ticket-router"
GITHUB_REMOTE = "https://github.com/kintpon/project-LLM.git"

OOS_INTENT = "oos"
HUMAN_TEAM = "Human agent"
