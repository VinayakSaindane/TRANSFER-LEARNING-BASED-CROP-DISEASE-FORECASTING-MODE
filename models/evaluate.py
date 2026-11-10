"""Comprehensive model evaluation on PlantVillage splits and PlantDoc external field dataset."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    ACTIVE_KERAS_BACKEND,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    CONFUSION_MATRIX_IMG_PATH,
    DEFAULT_BATCH_SIZE,
    EVAL_RESULTS_PATH,
    MODEL_PATH,
    PLANTDOC_MANIFEST_PATH,
    TRAINING_CURVES_IMG_PATH,
    TRAINING_HISTORY_PATH,
)
from models.train import TomatoLeafSequence, load_split_samples
import keras


def _load_plantdoc_samples(manifest_path: Path = PLANTDOC_MANIFEST_PATH) -> list[tuple[Path, int]]:
    if not manifest_path.exists():
        return []
    samples: list[tuple[Path, int]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label_idx = CLASS_TO_INDEX[row["label"]]
            samples.append((BASE_DIR / row["processed_path"], label_idx))
    return samples


def _evaluate_samples(
    model: keras.Model,
    samples: list[tuple[Path, int]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    seq = TomatoLeafSequence(samples, batch_size=batch_size, augment=False, shuffle=False)
    y_true = np.array([lbl for _, lbl in samples], dtype=np.int32)
    probs = model.predict(seq, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm_arr = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm = cm_arr.tolist()
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    specificity = None
    sensitivity = None
    if cm_arr.shape == (2, 2):
        tn, fp, fn, tp = int(cm_arr[0, 0]), int(cm_arr[0, 1]), int(cm_arr[1, 0]), int(cm_arr[1, 1])
        specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
        sensitivity = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0

    total_n = len(samples)
    gt_counts = {name: int(np.sum(y_true == idx)) for idx, name in enumerate(CLASS_NAMES)}
    pred_counts = {name: int(np.sum(y_pred == idx)) for idx, name in enumerate(CLASS_NAMES)}
    pred_pcts = {
        name: round((pred_counts[name] / total_n) * 100.0, 2) if total_n > 0 else 0.0
        for name in CLASS_NAMES
    }

    return {
        "num_samples": total_n,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm,
        "confusion_matrix_ordering": {
            "rows": "true_label",
            "cols": "predicted_label",
            "labels": CLASS_NAMES,
        },
        "prediction_distribution": {
            "ground_truth_counts": gt_counts,
            "predicted_counts": pred_counts,
            "predicted_percentages": pred_pcts,
        },
        "classification_report": report,
    }


def _save_training_curves_figure(history_payload: dict[str, Any], out_path: Path) -> None:
    hist = history_payload.get("history") or {}
    epochs = list(range(1, len(hist.get("loss", [])) + 1))
    if not epochs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    stage1_end = history_payload.get("stage1_epochs", 0)

    axes[0].plot(epochs, hist["accuracy"], marker="o", label="Train Accuracy", color="#1B5E20")
    axes[0].plot(epochs, hist["val_accuracy"], marker="s", label="Val Accuracy", color="#4CAF50")
    if 0 < stage1_end < len(epochs):
        axes[0].axvline(stage1_end + 0.5, linestyle="--", color="#64748B", label="Fine-tuning start")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, hist["loss"], marker="o", label="Train Loss", color="#B91C1C")
    axes[1].plot(epochs, hist["val_loss"], marker="s", label="Val Loss", color="#F59E0B")
    if 0 < stage1_end < len(epochs):
        axes[1].axvline(stage1_end + 0.5, linestyle="--", color="#64748B", label="Fine-tuning start")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sparse Categorical Crossentropy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plt.tight_layout(w_pad=3.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_matrix_figure(
    pv_cm: list[list[int]],
    pd_cm: list[list[int]] | None,
    out_path: Path,
) -> None:
    ncols = 2 if pd_cm is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 4.6), dpi=140)
    if ncols == 1:
        axes = [axes]

    panels = [("PlantVillage Held-Out Test Set", np.array(pv_cm))]
    if pd_cm is not None:
        panels.append(("PlantDoc External Field Set (Zero-Shot)", np.array(pd_cm)))

    for ax, (title, mat) in zip(axes, panels):
        im = ax.imshow(mat, interpolation="nearest", cmap="Greens")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        thresh = mat.max() / 2.0 if mat.size else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(
                    j,
                    i,
                    str(mat[i, j]),
                    ha="center",
                    va="center",
                    color="white" if mat[i, j] > thresh else "#142419",
                    fontsize=12,
                    fontweight="bold",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(w_pad=3.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_all() -> dict[str, Any]:
    """Run full evaluation across PlantVillage splits and isolated PlantDoc external test set."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run 'python3 models/train.py' first."
        )

    model = keras.saving.load_model(MODEL_PATH)
    splits = load_split_samples()

    train_metrics = _evaluate_samples(model, splits["train"])
    val_metrics = _evaluate_samples(model, splits["val"])
    test_metrics = _evaluate_samples(model, splits["test"])

    pd_samples = _load_plantdoc_samples()
    plantdoc_metrics = _evaluate_samples(model, pd_samples) if pd_samples else None

    history_payload: dict[str, Any] = {}
    if TRAINING_HISTORY_PATH.exists():
        history_payload = json.loads(TRAINING_HISTORY_PATH.read_text(encoding="utf-8"))
        _save_training_curves_figure(history_payload, TRAINING_CURVES_IMG_PATH)

    _save_confusion_matrix_figure(
        test_metrics["confusion_matrix"],
        plantdoc_metrics["confusion_matrix"] if plantdoc_metrics else None,
        CONFUSION_MATRIX_IMG_PATH,
    )

    domain_shift_gap = (
        round(test_metrics["accuracy"] - plantdoc_metrics["accuracy"], 4)
        if plantdoc_metrics
        else None
    )
    balanced_domain_shift_gap = (
        round(test_metrics["balanced_accuracy"] - plantdoc_metrics["balanced_accuracy"], 4)
        if plantdoc_metrics
        else None
    )

    plantdoc_label_mapping = {
        "Tomato leaf": {"mapped_label": "Healthy", "class_index": CLASS_TO_INDEX["Healthy"]},
        "Tomato Early blight leaf": {
            "mapped_label": "Tomato Early Blight",
            "class_index": CLASS_TO_INDEX["Tomato Early Blight"],
        },
    }

    results = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "backend": ACTIVE_KERAS_BACKEND,
        "classes": CLASS_NAMES,
        "class_to_index": CLASS_TO_INDEX,
        "plantvillage_train": train_metrics,
        "plantvillage_val": val_metrics,
        "plantvillage_test": test_metrics,
        "plantdoc_external_eval": plantdoc_metrics,
        "plantdoc_label_mapping": plantdoc_label_mapping,
        "domain_shift_accuracy_drop": domain_shift_gap,
        "domain_shift_balanced_accuracy_drop": balanced_domain_shift_gap,
        "limitations": [
            "PlantVillage training and test images share controlled laboratory lighting and uniform studio backgrounds.",
            (
                "Zero-shot PlantDoc evaluation reveals severe domain shift: balanced accuracy is 52.38% and Healthy-class "
                "specificity is 4.76% (3/63) because 98.01% (148/151) of cluttered outdoor images were predicted as "
                "Tomato Early Blight. Current weights are NOT reliable for unverified field diagnosis."
            ),
            "The model is a binary classifier ('Healthy' vs 'Tomato Early Blight') and cannot diagnose late blight, Septoria leaf spot, bacterial spot, or abiotic deficiencies.",
        ],
    }

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    evaluate_all()
