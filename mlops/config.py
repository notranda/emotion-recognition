from pathlib import Path

# Root project
PROJECT_ROOT = Path(".")

# =========================
# MODEL REGISTRY
# =========================
MODELS_ROOT = PROJECT_ROOT / "models"
ACTIVE_ROOT = MODELS_ROOT / "active"
ARCHIVE_ROOT = MODELS_ROOT / "archive"

# =========================
# DATA
# =========================
DATA_ROOT = PROJECT_ROOT / "data"
DATASET_ROOT_DEFAULT = DATA_ROOT / "datasetpreprocessed"
NEW_DATA_ROOT = DATA_ROOT / "new_data"

# =========================
# TRAINING DEFAULTS
# =========================
DEFAULT_IMG_SIZE = 48
DEFAULT_BATCH_SIZE = 64
DEFAULT_COLOR_MODE = "grayscale"

# =========================
# LABELS (FER2013 standard)
# =========================
DEFAULT_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]
