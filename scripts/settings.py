from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "gold-data-h4.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
LOG_DIR = ROOT_DIR / "logs"
STATE_PATH = ROOT_DIR / "data" / "bot_state.json"

SYMBOL = "GOLD"
START_DATE = datetime(2000, 1, 1)
LOT_SIZE = 1.5
