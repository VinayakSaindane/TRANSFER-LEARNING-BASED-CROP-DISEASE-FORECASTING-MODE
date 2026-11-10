"""Transfer learning training pipeline using MobileNetV2 (Keras 3)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    ACTIVE_KERAS_BACKEND,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_FINETUNE_LR,
    DEFAULT_HEAD_EPOCHS,
    DEFAULT_HEAD_LR,
    DROPOUT_RATE,
    FINETUNE_UNFREEZE_LAYERS,
    IMAGE_SIZE,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    SPLIT_MANIFEST_PATH,
    TRAINING_HISTORY_PATH,
)
from models.preprocessing import preprocess_rgb_array
import keras


class TomatoLeafSequence(keras.utils.PyDataset):
    """Memory-efficient PyDataset streaming 224x224 images batch-by-batch."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        augment: bool = False,
        shuffle: bool = False,
        seed: int = RANDOM_SEED,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.samples = list(samples)
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.indices = list(range(len(self.samples)))
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self) -> int:
        return math.ceil(len(self.samples) / self.batch_size)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _augment_image(self, img: Image.Image) -> Image.Image:
        if self.rng.random() < 0.5:
            img = ImageOps.mirror(img)
        if self.rng.random() < 0.5:
            img = ImageOps.flip(img)
        angle = self.rng.uniform(-18.0, 18.0)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
        if self.rng.random() < 0.7:
            img = ImageEnhance.Brightness(img).enhance(self.rng.uniform(0.88, 1.12))
        if self.rng.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(self.rng.uniform(0.90, 1.10))
        return img

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        x_batch = np.zeros((len(batch_indices), IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
        y_batch = np.zeros((len(batch_indices),), dtype=np.int32)

        for i, sample_idx in enumerate(batch_indices):
            path, label_idx = self.samples[sample_idx]
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                if rgb.size != IMAGE_SIZE:
                    rgb = rgb.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                if self.augment:
                    rgb = self._augment_image(rgb)
                x_batch[i] = preprocess_rgb_array(np.asarray(rgb, dtype=np.float32))
                y_batch[i] = label_idx

        return x_batch, y_batch


def load_split_samples(manifest_path: Path = SPLIT_MANIFEST_PATH) -> dict[str, list[tuple[Path, int]]]:
    """Read train, val, and test samples from the verified dataset split manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {manifest_path}. Run 'python3 data/prepare_datasets.py' first."
        )

    splits: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    with manifest_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split_name = row["split"]
            label_idx = CLASS_TO_INDEX[row["label"]]
            img_path = BASE_DIR / row["processed_path"]
            splits[split_name].append((img_path, label_idx))
    return splits


def build_mobilenetv2_classifier() -> tuple[keras.Model, keras.Model]:
    """Build MobileNetV2 transfer-learning architecture with ImageNet weights."""
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="leaf_image")
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = keras.layers.Dropout(DROPOUT_RATE, name="head_dropout")(x)
    outputs = keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="disease_classifier")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="AgriGuard_MobileNetV2")
    return model, base_model


def _extract_base_model(model: keras.Model) -> keras.Model | None:
    for layer in model.layers:
        if isinstance(layer, keras.Model) or "mobilenetv2" in layer.name.lower():
            return layer
    return None


def _compute_balanced_weights(train_samples: list[tuple[Path, int]]) -> dict[int, float]:
    y = np.array([lbl for _, lbl in train_samples], dtype=np.int32)
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(round(w, 4)) for c, w in zip(classes, weights)}


def _merge_histories(h1: dict[str, list[float]], h2: dict[str, list[float]]) -> dict[str, Any]:
    merged: dict[str, list[float]] = {}
    for key in ("loss", "val_loss", "accuracy", "val_accuracy"):
        merged[key] = [round(float(v), 6) for v in h1.get(key, [])] + [
            round(float(v), 6) for v in h2.get(key, [])
        ]
    return {
        "stage1_epochs": len(h1.get("loss", [])),
        "stage2_epochs": len(h2.get("loss", [])),
        "total_epochs": len(merged["loss"]),
        "history": merged,
    }


def train_model(
    head_epochs: int = DEFAULT_HEAD_EPOCHS,
    finetune_epochs: int = DEFAULT_FINETUNE_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = False,
    output_dir: Path | None = None,
    overwrite_best: bool = False,
) -> dict[str, Any]:
    """Train MobileNetV2 in two stages (frozen head -> top-layer fine-tuning).

    When ``resume=True``, training loads from ``CHECKPOINT_PATH`` and saves to a
    separate resumed artifact path (or ``output_dir``) by default so the best
    saved production model (``MODEL_PATH``) is never overwritten unless
    ``overwrite_best=True`` is explicitly requested.
    """
    keras.utils.set_random_seed(RANDOM_SEED)
    splits = load_split_samples()

    train_seq = TomatoLeafSequence(splits["train"], batch_size=batch_size, augment=True, shuffle=True)
    val_seq = TomatoLeafSequence(splits["val"], batch_size=batch_size, augment=False, shuffle=False)
    class_weights = _compute_balanced_weights(splits["train"])

    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        save_model_path = out_root / MODEL_PATH.name
        save_ckpt_path = out_root / CHECKPOINT_PATH.name
        save_history_path = out_root / TRAINING_HISTORY_PATH.name
        save_metadata_path = out_root / MODEL_METADATA_PATH.name
    elif resume and not overwrite_best:
        resumed_dir = MODEL_PATH.parent / "resumed"
        resumed_dir.mkdir(parents=True, exist_ok=True)
        save_model_path = resumed_dir / "agriguard_mobilenetv2_resumed.keras"
        save_ckpt_path = resumed_dir / "agriguard_checkpoint_resumed.keras"
        save_history_path = resumed_dir / "training_history_resumed.json"
        save_metadata_path = resumed_dir / "model_metadata_resumed.json"
    else:
        save_model_path = MODEL_PATH
        save_ckpt_path = CHECKPOINT_PATH
        save_history_path = TRAINING_HISTORY_PATH
        save_metadata_path = MODEL_METADATA_PATH

    baseline_val_loss: float | None = None
    if resume:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint not found at {CHECKPOINT_PATH}."
            )
        print(f"Resuming from checkpoint (read-only source): {CHECKPOINT_PATH}")
        model = keras.saving.load_model(CHECKPOINT_PATH)
        base_model = _extract_base_model(model)
        stage1_hist: dict[str, list[float]] = {}
        try:
            eval_out = model.evaluate(val_seq, verbose=0)
            baseline_val_loss = float(eval_out[0]) if isinstance(eval_out, (list, tuple)) else float(eval_out)
            print(f"Baseline checkpoint val_loss before resume: {baseline_val_loss:.5f}")
        except Exception:  # noqa: BLE001
            baseline_val_loss = None
    else:
        model, base_model = build_mobilenetv2_classifier()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=DEFAULT_HEAD_LR),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks_stage1 = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(save_ckpt_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]
        print(f"Stage 1: Training classification head for up to {head_epochs} epochs (backend={ACTIVE_KERAS_BACKEND})...")
        hist1_obj = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=head_epochs,
            class_weight=class_weights,
            callbacks=callbacks_stage1,
            verbose=1,
        )
        stage1_hist = hist1_obj.history

    stage2_hist: dict[str, list[float]] = {}
    if finetune_epochs > 0 and base_model is not None:
        base_model.trainable = True
        total_base_layers = len(base_model.layers)
        freeze_until = max(0, total_base_layers - FINETUNE_UNFREEZE_LAYERS)
        for idx, layer in enumerate(base_model.layers):
            if idx < freeze_until or isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=DEFAULT_FINETUNE_LR),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        ckpt_kwargs: dict[str, Any] = {
            "filepath": str(save_ckpt_path),
            "monitor": "val_loss",
            "save_best_only": True,
            "verbose": 1,
        }
        if baseline_val_loss is not None:
            ckpt_kwargs["initial_value_threshold"] = baseline_val_loss

        callbacks_stage2 = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(**ckpt_kwargs),
        ]
        print(
            f"Stage 2: Fine-tuning top {FINETUNE_UNFREEZE_LAYERS} non-BN layers for up to {finetune_epochs} epochs..."
        )
        hist2_obj = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=finetune_epochs,
            class_weight=class_weights,
            callbacks=callbacks_stage2,
            verbose=1,
        )
        stage2_hist = hist2_obj.history

    save_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_model_path)

    history_payload = _merge_histories(stage1_hist, stage2_hist)
    history_payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    history_payload["backend"] = ACTIVE_KERAS_BACKEND
    history_payload["batch_size"] = batch_size
    history_payload["class_weights"] = class_weights
    history_payload["resumed_from_checkpoint"] = bool(resume)
    history_payload["baseline_val_loss_before_resume"] = (
        round(baseline_val_loss, 6) if baseline_val_loss is not None else None
    )

    save_history_path.parent.mkdir(parents=True, exist_ok=True)
    with save_history_path.open("w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total_params = int(model.count_params())

    try:
        rel_model_path = str(save_model_path.relative_to(BASE_DIR))
    except ValueError:
        rel_model_path = str(save_model_path)

    metadata = {
        "model_name": "AgriGuard MobileNetV2",
        "backbone": "MobileNetV2 (ImageNet pretrained, include_top=False)",
        "keras_version": keras.__version__,
        "keras_backend": ACTIVE_KERAS_BACKEND,
        "input_shape": [IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
        "preprocessing": "keras.applications.mobilenet_v2.preprocess_input ([-1, 1] scaling)",
        "classes": CLASS_NAMES,
        "class_to_index": CLASS_TO_INDEX,
        "dropout_rate": DROPOUT_RATE,
        "head_learning_rate": DEFAULT_HEAD_LR,
        "finetune_learning_rate": DEFAULT_FINETUNE_LR,
        "finetune_unfrozen_upper_layers": FINETUNE_UNFREEZE_LAYERS,
        "total_parameters": total_params,
        "trainable_parameters_stage2": trainable_params,
        "saved_model_path": rel_model_path,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "resumed_from_checkpoint": bool(resume),
    }
    save_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with save_metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved trained model to {save_model_path}")
    return {"metadata": metadata, "history": history_payload, "model_path": str(save_model_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AgriGuard AI MobileNetV2 classifier.")
    parser.add_argument("--head-epochs", type=int, default=DEFAULT_HEAD_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for saving resumed or experimental artifacts without touching production files.",
    )
    parser.add_argument(
        "--overwrite-best",
        action="store_true",
        help="When resuming, allow overwriting models/saved/agriguard_mobilenetv2.keras.",
    )
    args = parser.parse_args()

    train_model(
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        resume=args.resume,
        output_dir=args.output_dir,
        overwrite_best=args.overwrite_best,
    )


if __name__ == "__main__":
    main()

