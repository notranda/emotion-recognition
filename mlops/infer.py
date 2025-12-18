from pathlib import Path
import json
import numpy as np
import cv2
import keras

from mlops.config import ACTIVE_ROOT, DEFAULT_LABELS

MODEL_DIRS = {
    "cnn": ACTIVE_ROOT / "cnn",
    "resnet50": ACTIVE_ROOT / "resnet50",
}

_model_cache = {}
_meta_cache = {}


def load_meta(model_key: str) -> dict:
    if model_key in _meta_cache:
        return _meta_cache[model_key]

    meta_path = MODEL_DIRS[model_key] / "meta.json"
    if not meta_path.exists():
        # fallback meta minimal
        meta = {
            "model_key": model_key,
            "version": "unknown",
            "img_size": 48,
            "color_mode": "grayscale",
            "labels": DEFAULT_LABELS,
        }
        _meta_cache[model_key] = meta
        return meta

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _meta_cache[model_key] = meta
    return meta

def _ensure_dirs():
    (ACTIVE_ROOT / "cnn").mkdir(parents=True, exist_ok=True)
    (ACTIVE_ROOT / "resnet50").mkdir(parents=True, exist_ok=True)
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    NEW_DATA_ROOT.mkdir(parents=True, exist_ok=True)


def load_model(model_key: str) -> keras.Model:
    model_path = ACTIVE_ROOT / model_key / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    return keras.saving.load_model(str(model_path))



def preprocess_frame(frame_bgr, model_key: str, img_size: int):
    if frame_bgr is None:
        raise ValueError("frame_bgr is None")

    # Convert to grayscale
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    if model_key == "cnn":
        # Resize the grayscale image for CNN
        resized = cv2.resize(gray, (img_size, img_size))
        x = resized.astype("float32") / 255.0  # Normalize pixel values
        x = np.expand_dims(x, axis=-1)  # Add the channel dimension for CNN (grayscale)
    else:
        # For ResNet50, convert to 3 channels (grayscale replicated as RGB)
        resized = cv2.resize(gray, (img_size, img_size))
        x = resized.astype("float32") / 255.0  # Normalize pixel values
        x = np.stack([x, x, x], axis=-1)  # Convert to RGB by stacking the same grayscale image into 3 channels

    x = np.expand_dims(x, axis=0)  # Add batch dimension
    return x



def predict(frame_bgr, model_key: str = "cnn"):
    if model_key not in MODEL_DIRS:
        raise ValueError(f"model_key must be one of {list(MODEL_DIRS.keys())}")

    meta = load_meta(model_key)
    model = load_model(model_key)

    img_size = int(meta.get("img_size", 48))
    labels = meta.get("labels", DEFAULT_LABELS)

    x = preprocess_frame(frame_bgr, model_key=model_key, img_size=img_size)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
    return label, conf, probs, meta