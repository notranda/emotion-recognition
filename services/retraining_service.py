from __future__ import annotations

from pathlib import Path
import json
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from mlops.config import (
    ACTIVE_ROOT,
    ARCHIVE_ROOT,
    NEW_DATA_ROOT,
    DATASET_ROOT_DEFAULT,
    DEFAULT_IMG_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LABELS,
)

def _ensure_dirs():
    (ACTIVE_ROOT / "cnn").mkdir(parents=True, exist_ok=True)
    (ACTIVE_ROOT / "resnet50").mkdir(parents=True, exist_ok=True)
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    NEW_DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _version_now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _archive_if_exists(model_key: str, version_new: str):
    src = ACTIVE_ROOT / model_key
    if not src.exists():
        return
    # archive only if it contains model
    if (src / "model.keras").exists():
        dst = ARCHIVE_ROOT / f"{model_key}_{version_new}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def _merge_new_data_into_dataset(dataset_root: Path) -> Path:
    """
    Merge data/new_data/<label> into a staging folder so retrain uses both.
    """
    staging = Path("data/_staging_merged")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    # copy dataset_root
    shutil.copytree(dataset_root, staging / "base", dirs_exist_ok=True)

    # merge new_data
    for label_dir in NEW_DATA_ROOT.glob("*"):
        if not label_dir.is_dir():
            continue
        target = staging / "base" / label_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for img in label_dir.glob("*.*"):
            shutil.copy2(img, target / img.name)

    return staging / "base"


def _build_cnn(img_size: int, num_classes: int):
    """
    CNN sederhana tapi cukup kuat.
    Input: (img_size,img_size,1) grayscale
    """
    inputs = keras.Input(shape=(img_size, img_size, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cnn_emotion")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_resnet50(img_size: int, num_classes: int):
    """
    ResNet50 transfer learning.
    Input: (img_size,img_size,3)
    NOTE: Kita akan train dengan grayscale-replicated RGB (pipeline preprocessing bisa atur).
    """
    base = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    # freeze most layers
    for layer in base.layers[:-10]:
        layer.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="resnet50_emotion")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def retrain(
    model_key: str = "cnn",
    epochs: int = 5,
    dataset_root: str | None = None,
    img_size: int | None = None,
    batch_size: int | None = None,
):
    """
    Train model (cnn/resnet50), save to models/active/<model_key>/
    Outputs:
      - model.keras
      - meta.json
      - version.txt
    """
    _ensure_dirs()

    if dataset_root is None:
        dataset_root_path = DATASET_ROOT_DEFAULT
    else:
        dataset_root_path = Path(dataset_root)

    if not dataset_root_path.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root_path}")

    img_size = img_size or DEFAULT_IMG_SIZE
    batch_size = batch_size or DEFAULT_BATCH_SIZE

    version = _version_now()

    # archive current active
    _archive_if_exists(model_key, version)

    # merge new_data
    merged_root = _merge_new_data_into_dataset(dataset_root_path)

    # datagen
    # untuk grayscale CNN -> color_mode grayscale
    # untuk resnet50 -> color_mode rgb (tapi bisa saja dataset grayscale; aman)
    if model_key == "cnn":
        color_mode = "grayscale"
        target_size = (img_size, img_size)
    else:
        color_mode = "rgb"
        target_size = (img_size, img_size)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    train_gen = train_datagen.flow_from_directory(
        directory=str(merged_root),
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        directory=str(merged_root),
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    num_classes = train_gen.num_classes

    # class weights (untuk imbalance)
    counts = {k: int(v) for k, v in train_gen.classes.tolist().__class__ == list and []}  # dummy to satisfy lint
    # compute from generator
    unique, freq = np.unique(train_gen.classes, return_counts=True)
    class_weight = {}
    total = float(np.sum(freq))
    for cls_id, f in zip(unique, freq):
        class_weight[int(cls_id)] = total / (len(unique) * float(f))

    # build model
    if model_key == "cnn":
        model = _build_cnn(img_size=img_size, num_classes=num_classes)
    elif model_key == "resnet50":
        model = _build_resnet50(img_size=img_size, num_classes=num_classes)
    else:
        raise ValueError("model_key must be 'cnn' or 'resnet50'")

    # callbacks
    out_dir = ACTIVE_ROOT / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "best.keras"
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # save final active model
    model_path = out_dir / "model.keras"
    keras.saving.save_model(model, str(model_path))  # Keras 3 safe save

    # meta
    labels = list(train_gen.class_indices.keys())
    meta = {
        "model_key": model_key,
        "version": version,
        "img_size": img_size,
        "color_mode": "grayscale" if model_key == "cnn" else "rgb",
        "labels": labels,
        "class_indices": train_gen.class_indices,
        "trained_at": datetime.now().isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "version.txt").write_text(version, encoding="utf-8")

    # return info
    return {
        "model_key": model_key,
        "version": version,
        "img_size": img_size,
        "classes": labels,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "saved_to": str(out_dir),
    }
