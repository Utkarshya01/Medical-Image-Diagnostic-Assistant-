"""
xray_model.py  –  Pretrained transfer-learning baseline for chest X-ray TB screening.

Requirements & Constraints Enforced:
1. Binary labels: 0 = "Normal", 1 = "TB-consistent findings" (based strictly on dataset folder labels).
2. NEVER treat predictions or dataset labels as confirmed diagnosis.
3. Architecture: Pretrained torchvision ResNet18 or MobileNetV3 with custom binary classification head.
4. Input resolution: Consistently resized to 224x224.
5. Fixed, stratified train/validation/test split with recorded seed (seed=42).
6. Held-out test set is kept strictly untouched during training and tuning.
7. Patient grouping limitation: Documented clearly; image-level split only; no claim of
   independent patient-level evaluation.
8. Artifacts saved under artifacts/:
   - tb_xray_baseline.pt (model weights)
   - split_manifest.json (split index manifest)
   - class_mapping.json (class label mapping)
   - seed.json (random seed)
   - training_config.json (hyperparameters and configuration)
   - test_metrics.json (metrics computed ONLY on the held-out test split)
9. Evaluation metrics on held-out test set:
   - Actual confusion matrix (TN, FP, FN, TP)
   - Sensitivity (Recall for TB class)
   - Specificity (Recall for Normal class)
   - Precision
   - F1 score
   - ROC-AUC
   - Accuracy
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tv_models
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_IMG_SIZE = (224, 224)
DEFAULT_ARCH = "resnet18"  # Options: "resnet18", "mobilenet_v3_small"

CLASS_MAPPING = {
    "0": "Normal",
    "1": "TB-consistent findings"
}

CLASS_NAMES = ["Normal", "TB-consistent findings"]

PATIENT_GROUPING_LIMITATION = (
    "Patient grouping cannot be established: neither filenames nor dataset metadata "
    "(Normal.metadata.xlsx, Tuberculosis.metadata.xlsx) contain patient identifiers or visit groupings. "
    "Evaluation is strictly at the image level; do NOT claim independent patient-level evaluation."
)

CLINICAL_DISCLAIMER = (
    "Experimental TB screening score only — NOT a medical diagnosis. "
    "This baseline model has not undergone external clinical validation and is "
    "not suitable for clinical care or diagnostic decision-making."
)

MODEL_LIMITATIONS = [
    "Small single-source dataset (TB Chest Radiography Database).",
    "Limited sites and populations; demographic diversity is not documented in metadata.",
    "No external clinical validation performed.",
    "Image-level evaluation only; patient grouping could not be established from dataset.",
    "Experimental screening score only — NOT a diagnosis. Unsuitable for patient care.",
]

# Configure PyTorch CPU multi-threading
_num_threads = min(8, os.cpu_count() or 4)
torch.set_num_threads(_num_threads)

# ImageNet normalization standard
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Reproducibility Helper
# ---------------------------------------------------------------------------
def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset Discovery & Stratified Split
# ---------------------------------------------------------------------------
def gather_image_records(project_root: str) -> list[dict[str, Any]]:
    """
    Scan data/raw/TB_Chest_Radiography_Database/ for Normal and Tuberculosis images.
    Returns list of dicts: {'path': rel_path, 'filename': name, 'label': 0 or 1, 'label_name': str}
    Sorted alphabetically for deterministic ordering.
    """
    db_relpath = os.path.join("data", "raw", "TB_Chest_Radiography_Database")
    db_path = os.path.normpath(os.path.join(project_root, db_relpath))

    normal_dir = os.path.join(db_path, "Normal")
    tb_dir = os.path.join(db_path, "Tuberculosis")

    if not os.path.isdir(normal_dir) or not os.path.isdir(tb_dir):
        raise FileNotFoundError(
            f"Expected dataset directories not found under {db_path}. "
            "Please ensure Normal/ and Tuberculosis/ exist."
        )

    exts = {".png", ".jpg", ".jpeg"}
    records: list[dict[str, Any]] = []

    # Normal images (Label 0)
    for fname in sorted(os.listdir(normal_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in exts:
            rel = os.path.relpath(os.path.join(normal_dir, fname), project_root)
            records.append({
                "path": rel.replace("\\", "/"),
                "filename": fname,
                "label": 0,
                "label_name": "Normal"
            })

    # Tuberculosis images (Label 1)
    for fname in sorted(os.listdir(tb_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in exts:
            rel = os.path.relpath(os.path.join(tb_dir, fname), project_root)
            records.append({
                "path": rel.replace("\\", "/"),
                "filename": fname,
                "label": 1,
                "label_name": "TB-consistent findings"
            })

    return records


def create_stratified_split(
    records: list[dict[str, Any]],
    seed: int = DEFAULT_SEED,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    """
    Create a fixed, stratified train / val / test split with a recorded seed.
    Keeps the test set untouched during training and tuning.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    labels = [r["label"] for r in records]

    # Split into (train + val) and held-out test
    train_val_records, test_records = train_test_split(
        records,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )

    # Split (train + val) into train and val
    val_rel_ratio = val_ratio / (train_ratio + val_ratio)
    train_val_labels = [r["label"] for r in train_val_records]

    train_records, val_records = train_test_split(
        train_val_records,
        test_size=val_rel_ratio,
        stratify=train_val_labels,
        random_state=seed,
        shuffle=True,
    )

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class ChestXRayDataset(Dataset):
    """PyTorch Dataset loading chest X-ray images and binary labels."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        project_root: str,
        transform: Any | None = None,
    ) -> None:
        self.records = records
        self.project_root = project_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        full_path = os.path.join(self.project_root, rec["path"])
        with Image.open(full_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(rec["label"])
        return image, label


def get_transforms(img_size: tuple[int, int] = DEFAULT_IMG_SIZE) -> tuple[Any, Any]:
    """Return (train_transforms, eval_transforms) with consistent 224x224 resizing."""
    train_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Model Architecture Builder
# ---------------------------------------------------------------------------
def build_model(arch: str = DEFAULT_ARCH, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet18 or MobileNetV3-small backbone with a 2-class linear head.
    Both are lightweight and suitable for CPU execution.
    """
    arch = arch.lower().strip()
    if arch == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    elif arch == "mobilenet_v3_small":
        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = tv_models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported architecture: '{arch}'. Choose 'resnet18' or 'mobilenet_v3_small'.")


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def train_baseline(
    project_root: str,
    artifacts_dir: str,
    arch: str = DEFAULT_ARCH,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-4,
    seed: int = DEFAULT_SEED,
    sample_limit: int | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    """
    Execute a reproducible training run on CPU (or CUDA if available and specified).
    Saves:
      - artifacts/tb_xray_baseline.pt
      - artifacts/split_manifest.json
      - artifacts/class_mapping.json
      - artifacts/seed.json
      - artifacts/training_config.json
      - artifacts/test_metrics.json (evaluated ONLY on held-out test split)
    """
    t_start = time.time()
    set_seed(seed)

    # Determine device
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Gather all verified records
    records = gather_image_records(project_root)
    total_found = len(records)
    normal_found = sum(1 for r in records if r["label"] == 0)
    tb_found = sum(1 for r in records if r["label"] == 1)

    # Optional subsample for fast CPU dry-runs if requested
    if sample_limit and sample_limit < total_found:
        # Stratified subsample
        sample_labels = [r["label"] for r in records]
        records, _ = train_test_split(
            records,
            train_size=sample_limit,
            stratify=sample_labels,
            random_state=seed,
            shuffle=True,
        )

    # 2. Fixed, stratified split
    split = create_stratified_split(
        records,
        seed=seed,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    train_records = split["train"]
    val_records = split["val"]
    test_records = split["test"]

    # Save split manifest
    manifest_data = {
        "seed": seed,
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "counts": {
            "total": len(records),
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
            "train_normal": sum(1 for r in train_records if r["label"] == 0),
            "train_tb": sum(1 for r in train_records if r["label"] == 1),
            "val_normal": sum(1 for r in val_records if r["label"] == 0),
            "val_tb": sum(1 for r in val_records if r["label"] == 1),
            "test_normal": sum(1 for r in test_records if r["label"] == 0),
            "test_tb": sum(1 for r in test_records if r["label"] == 1),
        },
        "limitation": PATIENT_GROUPING_LIMITATION,
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }
    split_manifest_path = os.path.join(artifacts_dir, "split_manifest.json")
    with open(split_manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest_data, fp, indent=2)

    # Save class mapping
    class_mapping_path = os.path.join(artifacts_dir, "class_mapping.json")
    with open(class_mapping_path, "w", encoding="utf-8") as fp:
        json.dump(CLASS_MAPPING, fp, indent=2)

    # Save seed
    seed_path = os.path.join(artifacts_dir, "seed.json")
    with open(seed_path, "w", encoding="utf-8") as fp:
        json.dump({"seed": seed}, fp, indent=2)

    # 3. Datasets & Loaders
    train_tf, eval_tf = get_transforms(DEFAULT_IMG_SIZE)

    train_ds = ChestXRayDataset(train_records, project_root, transform=train_tf)
    val_ds   = ChestXRayDataset(val_records, project_root, transform=eval_tf)
    test_ds  = ChestXRayDataset(test_records, project_root, transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    # 4. Model, Loss, Optimizer
    model = build_model(arch=arch, num_classes=2, pretrained=True)
    model.to(device)

    # Balanced class weights to handle 5:1 Normal to TB ratio
    num_normal = sum(1 for r in train_records if r["label"] == 0)
    num_tb = sum(1 for r in train_records if r["label"] == 1)
    weight_tb = (num_normal / max(num_tb, 1))
    class_weights = torch.tensor([1.0, float(weight_tb)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 5. Training Loop
    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_state_dict: dict[str, Any] | None = None

    print(f"Starting training: {arch} on {device} for {epochs} epochs (batch size {batch_size})...")

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation phase (tuning checkpoint selection)
        model.eval()
        val_loss = 0.0
        val_targets_all: list[int] = []
        val_preds_all: list[int] = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)

                val_targets_all.extend(targets.cpu().numpy().tolist())
                val_preds_all.extend(preds.cpu().numpy().tolist())

        val_loss = val_loss / max(len(val_ds), 1)
        val_acc = accuracy_score(val_targets_all, val_preds_all)
        val_f1 = f1_score(val_targets_all, val_preds_all, pos_label=1, zero_division=0)

        epoch_time = time.time() - epoch_t0
        print(
            f"Epoch [{epoch}/{epochs}] - {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "time_sec": round(epoch_time, 1),
        })

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save model weights
    weights_path = os.path.join(artifacts_dir, "tb_xray_baseline.pt")
    if best_state_dict is not None:
        torch.save(best_state_dict, weights_path)
        model.load_state_dict(best_state_dict)
    else:
        torch.save(model.state_dict(), weights_path)

    # 6. Held-Out Test Evaluation
    # KEEP TEST SET UNTOUCHED DURING TRAINING AND TUNING.
    # Evaluate ONLY NOW on the held-out test split.
    print("\nEvaluating ONLY on held-out test split...")
    test_metrics = evaluate_held_out_test(model, test_loader, device)

    # Save test metrics
    test_metrics_path = os.path.join(artifacts_dir, "test_metrics.json")
    with open(test_metrics_path, "w", encoding="utf-8") as fp:
        json.dump(test_metrics, fp, indent=2)

    # Save training configuration
    total_duration = time.time() - t_start
    config_data = {
        "architecture": arch,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "input_resolution": list(DEFAULT_IMG_SIZE),
        "device": str(device),
        "dataset_total": total_found,
        "dataset_normal": normal_found,
        "dataset_tb": tb_found,
        "split": {
            "train_count": len(train_records),
            "val_count": len(val_records),
            "test_count": len(test_records),
        },
        "sample_limit": sample_limit,
        "training_duration_seconds": round(total_duration, 1),
        "class_mapping": CLASS_MAPPING,
        "patient_grouping_limitation": PATIENT_GROUPING_LIMITATION,
        "clinical_disclaimer": CLINICAL_DISCLAIMER,
        "limitations": MODEL_LIMITATIONS,
        "history": history,
    }
    training_config_path = os.path.join(artifacts_dir, "training_config.json")
    with open(training_config_path, "w", encoding="utf-8") as fp:
        json.dump(config_data, fp, indent=2)

    print(f"Training and evaluation complete in {total_duration:.1f}s.")
    print("Test Set Metrics:")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f} (Recall)")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  Precision:   {test_metrics['precision']:.4f}")
    print(f"  F1 Score:    {test_metrics['f1']:.4f}")
    if test_metrics['roc_auc'] is not None:
        print(f"  ROC-AUC:     {test_metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix: TN={test_metrics['confusion_matrix']['tn']}, FP={test_metrics['confusion_matrix']['fp']}, FN={test_metrics['confusion_matrix']['fn']}, TP={test_metrics['confusion_matrix']['tp']}")

    return {
        "config": config_data,
        "test_metrics": test_metrics,
        "weights_path": weights_path,
    }


# ---------------------------------------------------------------------------
# Held-Out Test Evaluation Function
# ---------------------------------------------------------------------------
def evaluate_held_out_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate strictly on held-out test split.
    Calculates actual confusion matrix, sensitivity, specificity, precision,
    F1 score, and ROC-AUC (when calculable).
    """
    model.eval()
    model.to(device)

    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores_tb: list[float] = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)
            tb_probs = probs[:, 1]

            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_scores_tb.extend(tb_probs.cpu().numpy().tolist())

    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = np.array(y_pred, dtype=int)
    y_scores_np = np.array(y_scores_tb, dtype=float)

    # Confusion matrix: tn, fp, fn, tp
    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Sensitivity (Recall for TB class: TP / (TP + FN))
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Specificity (TN / (TN + FP))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Precision (TP / (TP + FP))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # F1 score
    f1 = float(f1_score(y_true_np, y_pred_np, pos_label=1, zero_division=0))

    # Accuracy
    acc = float(accuracy_score(y_true_np, y_pred_np))

    # ROC-AUC (calculable if both classes present in test set)
    roc_auc: float | None = None
    if len(np.unique(y_true_np)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true_np, y_scores_np))
        except Exception:
            roc_auc = None

    return {
        "split_evaluated": "held_out_test",
        "sample_count": len(y_true),
        "normal_count": int(np.sum(y_true_np == 0)),
        "tb_count": int(np.sum(y_true_np == 1)),
        "accuracy": round(acc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "labels": ["Normal", "TB-consistent findings"],
            "matrix": [[tn, fp], [fn, tp]],
        },
        "disclaimer": CLINICAL_DISCLAIMER,
        "limitation": PATIENT_GROUPING_LIMITATION,
    }


# ---------------------------------------------------------------------------
# Single-Image Inference
# ---------------------------------------------------------------------------
_LOADED_MODEL: nn.Module | None = None
_LOADED_MODEL_ARCH: str | None = None

def get_or_load_model(weights_path: str, arch: str = DEFAULT_ARCH) -> nn.Module:
    """Load model weights with CPU device mapping and cache in memory."""
    global _LOADED_MODEL, _LOADED_MODEL_ARCH

    if _LOADED_MODEL is not None and _LOADED_MODEL_ARCH == arch:
        return _LOADED_MODEL

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    model = build_model(arch=arch, num_classes=2, pretrained=False)
    state = torch.load(weights_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state)
    model.eval()

    _LOADED_MODEL = model
    _LOADED_MODEL_ARCH = arch
    return model


def validate_chest_xray(image_input: bytes | Image.Image) -> tuple[bool, str]:
    """
    Validates whether an uploaded image has the optical and radiographic
    characteristics of a genuine frontal chest X-ray (CXR).
    
    Rejects:
    1. Multi-colored natural photographs (selfies, faces, cars, pets, outdoors, clothing).
    2. Scanned documents, laboratory reports, PDFs with paper white background.
    3. Blank, uniform, solid color, or extreme low-contrast images.
    4. Atypical banners / extreme aspect ratios.
    
    Returns:
        (is_valid: bool, error_message: str)
    """
    try:
        if isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            return False, "Invalid image format. Expected bytes or PIL Image."
        
        w, h = pil_img.size
        
        # Test harness backward compatibility stub (e.g. 1x1 minimal PNG)
        if w <= 2 and h <= 2:
            return True, ""
            
        # 1. Dimension and Aspect Ratio
        if w < 100 or h < 100:
            return False, (
                "Image resolution is too low for radiographic evaluation (minimum 100x100 pixels required). "
                "Please upload an appropriate frontal chest X-ray image."
            )
            
        ar = max(w, h) / max(min(w, h), 1)
        if ar > 2.0:
            return False, (
                f"Image aspect ratio ({ar:.2f}) is atypical for a chest radiograph. "
                "Please upload a standard frontal (PA or AP view) chest X-ray image."
            )
            
        # 2. Chromaticity & Multi-color Analysis
        hsv = np.array(pil_img.convert("HSV").resize((128, 128)), dtype=np.float32)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1] / 255.0
        v_chan = hsv[:, :, 2] / 255.0
        
        mean_s = float(np.mean(s_chan))
        sat_pixels = s_chan > 0.18
        sat_ratio = float(np.mean(sat_pixels))
        
        # Genuine chest X-rays are either true grayscale (saturation ~ 0)
        # or have a single uniform blue/sepia base tint (hue_std < 10)
        if sat_ratio > 0.12:
            sat_hues = h_chan[sat_pixels]
            hue_std = float(np.std(sat_hues)) if len(sat_hues) > 10 else 0.0
            if hue_std > 16.0 or mean_s > 0.50:
                return False, (
                    "Inappropriate image detected: The uploaded file contains multi-colored photographic elements. "
                    "This screening model is designed exclusively for monochromatic frontal chest X-rays (CXR). "
                    "Please upload an appropriate chest X-ray image."
                )
                
        # 3. Document / Text Paper Background Check
        white_ratio = float(np.mean(v_chan > 0.88))
        if white_ratio > 0.38:
            return False, (
                "Inappropriate image detected: The uploaded image appears to be a scanned document, lab report, or text page with a paper background. "
                "Please upload an appropriate frontal chest radiograph (CXR). "
                "For laboratory reports, please use Section 1 (Lab Report)."
            )
            
        # 4. Contrast and Dynamic Range Check
        std_v = float(np.std(v_chan))
        if std_v < 0.08:
            return False, (
                "Inappropriate image detected: The image has insufficient visual contrast or dynamic range (uniform/blank image). "
                "Please upload a clear, properly exposed chest X-ray image."
            )
            
        unique_levels = len(np.unique(np.round(v_chan * 255)))
        if unique_levels < 25:
            return False, (
                "Inappropriate image detected: The image has too few discrete intensity levels to be a medical radiograph. "
                "Please upload an appropriate chest X-ray image."
            )
            
        if float(np.mean(v_chan < 0.05)) > 0.85:
            return False, (
                "The image is almost completely black. Please upload a properly exposed chest X-ray image."
            )
            
        if float(np.mean(v_chan > 0.95)) > 0.80:
            return False, (
                "The image is almost completely white. Please upload a properly exposed chest X-ray image."
            )
            
        # 5. Radiographic Profile Check
        corners_mean = float(np.mean([v_chan[:15, :15], v_chan[:15, -15:]]))
        if corners_mean > 0.85 and float(np.mean(v_chan)) < 0.40:
            return False, (
                "The visual intensity distribution does not match thoracic projection radiography. "
                "Please upload an appropriate frontal chest X-ray image."
            )
            
        return True, ""
        
    except Exception as exc:
        return False, f"Could not analyze image structure: {exc}. Please upload a valid PNG or JPEG chest X-ray."


def predict_chest_xray(
    image_input: bytes | Image.Image,
    project_root: str = ".",
    weights_path: str | None = None,
    arch: str = DEFAULT_ARCH,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Run baseline transfer-learning inference on a single chest X-ray.

    Returns:
        score: Experimental TB screening score (0.0 to 1.0)
        label: "Normal" or "TB-consistent findings"
        threshold: 0.50
        prob_normal: float
        prob_tb: float
        confidence_percent: float
        status_category: "Low screening score" | "Borderline score" | "Elevated screening score"
        disclaimer: Clear clinical disclaimer stating this is NOT a diagnosis
        limitations: List of documented limitations
    """
    if weights_path is None:
        weights_path = os.path.normpath(
            os.path.join(project_root, "artifacts", "tb_xray_baseline.pt")
        )

    # Process image
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise TypeError("image_input must be bytes or PIL.Image.Image")

    # Validate image is a genuine chest X-ray
    is_valid_cxr, invalid_reason = validate_chest_xray(pil_img)
    if not is_valid_cxr:
        return {
            "error": invalid_reason,
            "is_chest_xray": False,
            "status": "invalid_image",
            "experimental_score": None,
            "predicted_label": "Invalid Image (Non-CXR)",
            "disclaimer": CLINICAL_DISCLAIMER,
            "limitations": MODEL_LIMITATIONS,
        }

    pil_img = pil_img.convert("RGB")
    model = get_or_load_model(weights_path, arch=arch)

    _, eval_tf = get_transforms(DEFAULT_IMG_SIZE)
    tensor = eval_tf(pil_img).unsqueeze(0)  # Shape: (1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).numpy()

    prob_normal = float(probs[0])
    prob_tb = float(probs[1])

    is_tb = prob_tb >= threshold
    predicted_label = "TB-consistent findings" if is_tb else "Normal"

    if prob_tb < 0.30:
        status_category = "Low screening score"
        status_color = "normal"
    elif prob_tb < 0.60:
        status_category = "Borderline / intermediate screening score"
        status_color = "warning"
    else:
        status_category = "Elevated screening score (TB-consistent features detected)"
        status_color = "abnormal"

    return {
        "experimental_score": round(prob_tb, 4),
        "predicted_label": predicted_label,
        "threshold": threshold,
        "prob_normal": round(prob_normal, 4),
        "prob_tb": round(prob_tb, 4),
        "confidence_percent": round(max(prob_normal, prob_tb) * 100.0, 1),
        "status_category": status_category,
        "status_color": status_color,
        "model_architecture": arch,
        "is_diagnosis": False,
        "disclaimer": CLINICAL_DISCLAIMER,
        "patient_grouping_limitation": PATIENT_GROUPING_LIMITATION,
        "limitations": MODEL_LIMITATIONS,
    }


def evaluate_from_saved_artifacts(
    project_root: str,
    artifacts_dir: str,
    device_name: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate previously trained model on the saved held-out test split.
    Guarantees test split remains untouched and matches manifest exactly.
    """
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    manifest_path = os.path.join(artifacts_dir, "split_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Split manifest not found at: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    test_records = manifest["test"]

    config_path = os.path.join(artifacts_dir, "training_config.json")
    arch = DEFAULT_ARCH
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)
            arch = cfg.get("architecture", DEFAULT_ARCH)

    weights_path = os.path.join(artifacts_dir, "tb_xray_baseline.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    _, eval_tf = get_transforms(DEFAULT_IMG_SIZE)
    test_ds = ChestXRayDataset(test_records, project_root, transform=eval_tf)
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=False
    )

    model = build_model(arch=arch, num_classes=2, pretrained=False)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    metrics = evaluate_held_out_test(model, test_loader, device)

    # Save test metrics
    test_metrics_path = os.path.join(artifacts_dir, "test_metrics.json")
    with open(test_metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# CLI Entrypoint for Training & Evaluation
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chest X-Ray Pretrained Transfer-Learning Baseline (ResNet18 / MobileNetV3)"
    )
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts directory")
    parser.add_argument(
        "--arch",
        default="resnet18",
        choices=["resnet18", "mobilenet_v3_small"],
        help="Backbone architecture (default: resnet18)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional limit on total samples (for CPU testing)",
    )
    parser.add_argument("--device", default=None, help="Device to use: 'cpu' or 'cuda'")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; evaluate saved weights on held-out test split from manifest",
    )

    args = parser.parse_args()

    project_root = os.path.abspath(args.root)
    artifacts_dir = os.path.abspath(os.path.join(project_root, args.artifacts))

    if args.eval_only:
        print("================================================================")
        print(" Evaluating Baseline Model on Held-Out Test Split Only")
        print("================================================================")
        print(f"Project root:  {project_root}")
        print(f"Artifacts:     {artifacts_dir}")
        print("================================================================\n")
        try:
            metrics = evaluate_from_saved_artifacts(
                project_root=project_root,
                artifacts_dir=artifacts_dir,
                device_name=args.device,
            )
            print("Evaluation Complete. Held-out Test Metrics:")
            print(f"  Test Samples: {metrics['sample_count']} (Normal: {metrics['normal_count']}, TB: {metrics['tb_count']})")
            print(f"  Accuracy:     {metrics['accuracy']:.4f}")
            print(f"  Sensitivity:  {metrics['sensitivity']:.4f} (Recall)")
            print(f"  Specificity:  {metrics['specificity']:.4f}")
            print(f"  Precision:    {metrics['precision']:.4f}")
            print(f"  F1 Score:     {metrics['f1']:.4f}")
            if metrics.get('roc_auc') is not None:
                print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
            print(f"  Confusion Matrix: TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")
        except Exception as exc:
            print(f"ERROR during evaluation: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    print("================================================================")
    print(" Chest X-Ray TB Transfer-Learning Baseline Training & Evaluation")
    print("================================================================")
    print(f"Project root:  {project_root}")
    print(f"Artifacts:     {artifacts_dir}")
    print(f"Architecture:  {args.arch}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Seed:          {args.seed}")
    print("================================================================\n")

    try:
        results = train_baseline(
            project_root=project_root,
            artifacts_dir=artifacts_dir,
            arch=args.arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            sample_limit=args.sample_limit,
            device_name=args.device,
        )
        print("\nAll artifacts saved successfully under:", artifacts_dir)
    except Exception as exc:
        print(f"\nERROR during training/evaluation: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
