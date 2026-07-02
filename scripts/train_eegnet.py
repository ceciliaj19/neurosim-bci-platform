"""EEGNet training pipeline for motor imagery classification.

Usage
-----
    python scripts/train_eegnet.py [--epochs N] [--batch-size N] [--lr F]

Outputs
-------
models/eegnet_motor_imagery_v1.pth
    Full model checkpoint (best validation accuracy).
results/metrics.json
    Final test-set metrics: accuracy, precision, recall, F1.
results/training_history.json
    Per-epoch training and validation loss / accuracy.
results/training_curve.png
    Loss and accuracy curves over epochs.
results/confusion_matrix.json
    Raw confusion matrix as ``{"labels": [0, 1], "matrix": [[...], [...]]}``.
results/confusion_matrix.png
    Annotated confusion matrix heatmap figure.
"""

from __future__ import annotations

# Use non-interactive backend before any other matplotlib import so the
# script runs safely on headless servers with no display.
import matplotlib
matplotlib.use("Agg")

import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Fail fast with a clear message when PyTorch is absent
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required to run this script.\n"
        "Install it with:  pip install neurosim[training]\n"
        "or:               pip install torch"
    ) from exc

from neurosim.datasets import DatasetLoader
from neurosim.networks import EEGNet

# ---------------------------------------------------------------------------
# Project-relative output paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH  = _ROOT / "models"   / "eegnet_motor_imagery_v1.pth"
_METRICS_PATH = _ROOT / "results" / "metrics.json"
_HISTORY_PATH = _ROOT / "results" / "training_history.json"
_CURVE_PATH   = _ROOT / "results" / "training_curve.png"
_CM_JSON_PATH = _ROOT / "results" / "confusion_matrix.json"
_CM_PNG_PATH  = _ROOT / "results" / "confusion_matrix.png"


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

class _Split(NamedTuple):
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the motor imagery dataset via DatasetLoader.

    Returns
    -------
    X : np.ndarray
        EEG trials, shape ``(n_trials, n_channels, n_timepoints)``.
    y : np.ndarray
        Integer class labels, shape ``(n_trials,)``.
    """
    print("Loading dataset …")
    loader = DatasetLoader()
    X, y = loader.load_motor_imagery()
    print(f"  X: {X.shape}  y: {y.shape}  classes: {sorted(set(y.tolist()))}")
    return X, y


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> _Split:
    """Stratified train/test split using NumPy (no scikit-learn required).

    Parameters
    ----------
    X : np.ndarray
        Feature array, shape ``(n_samples, ...)``.
    y : np.ndarray
        Integer class labels, shape ``(n_samples,)``.
    test_size : float, optional
        Fraction of samples to hold out. Default ``0.2``.
    seed : int, optional
        Random seed for reproducibility. Default ``42``.

    Returns
    -------
    _Split
        Named tuple of ``(X_train, X_test, y_train, y_test)``.
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_size))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    print(
        f"Split — train: {len(train_idx)} samples  "
        f"test: {len(test_idx)} samples"
    )
    return _Split(
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
    )


def make_loaders(
    split: _Split,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Wrap numpy arrays in PyTorch DataLoaders.

    Parameters
    ----------
    split : _Split
        Output of :func:`stratified_split`.
    batch_size : int
        Mini-batch size for both loaders.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    """
    def _to_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long(),
        )

    train_ds = _to_tensor_dataset(split.X_train, split.y_train)
    val_ds = _to_tensor_dataset(split.X_test, split.y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one full training epoch.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    criterion : nn.Module
        Loss function.
    optimiser : torch.optim.Optimizer
    device : torch.device

    Returns
    -------
    loss : float
        Mean loss over all batches.
    accuracy : float
        Fraction of correctly classified samples.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        n += len(y_batch)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a DataLoader.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    criterion : nn.Module
    device : torch.device

    Returns
    -------
    loss : float
    accuracy : float
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        n += len(y_batch)

    return total_loss / n, correct / n


@torch.no_grad()
def compute_test_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int = 2,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Compute accuracy, per-class precision, recall, and macro F1.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device
    n_classes : int, optional
        Number of classes. Default ``2``.

    Returns
    -------
    metrics : dict[str, float]
        Keys: ``accuracy``, ``precision_macro``, ``recall_macro``, ``f1_macro``.
    all_preds : np.ndarray
        Predicted class indices for every test sample, shape ``(n_test,)``.
    all_labels : np.ndarray
        Ground-truth class indices for every test sample, shape ``(n_test,)``.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for X_batch, y_batch in loader:
        preds = model(X_batch.to(device)).argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y_batch.tolist())

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)

    accuracy = float((preds_arr == labels_arr).mean())

    precisions, recalls, f1s = [], [], []
    for cls in range(n_classes):
        tp = int(((preds_arr == cls) & (labels_arr == cls)).sum())
        fp = int(((preds_arr == cls) & (labels_arr != cls)).sum())
        fn = int(((preds_arr != cls) & (labels_arr == cls)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    metrics = {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
    }
    return metrics, preds_arr, labels_arr


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    n_epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    """Run the full training pipeline.

    Parameters
    ----------
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Initial learning rate for Adam.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Data ----------------------------------------------------------------
    X, y = load_data()
    split = stratified_split(X, y)
    train_loader, val_loader = make_loaders(split, batch_size)

    # -- Model ---------------------------------------------------------------
    n_channels, n_timepoints = X.shape[1], X.shape[2]
    n_classes = len(np.unique(y))

    model = EEGNet(
        n_classes=n_classes,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EEGNet — {n_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs
    )

    # -- Training loop -------------------------------------------------------
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_val_acc = 0.0
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimiser, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(
            f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  "
            f"{vl_loss:>9.4f}  {vl_acc:>8.4f}  {elapsed:>5.1f}s"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), _MODEL_PATH)
            print(f"         ✓ saved best model (val_acc={best_val_acc:.4f})")

    # -- Test metrics --------------------------------------------------------
    print("\nEvaluating best checkpoint on test set …")
    best_model = EEGNet(
        n_classes=n_classes,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
    ).to(device)
    best_model.load_state_dict(
        torch.load(_MODEL_PATH, map_location=device, weights_only=True)
    )
    metrics, all_preds, all_labels = compute_test_metrics(
        best_model, val_loader, device, n_classes
    )
    metrics["best_val_acc"] = best_val_acc
    metrics["n_epochs"] = n_epochs
    metrics["batch_size"] = batch_size
    metrics["lr"] = lr

    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # -- Save results --------------------------------------------------------
    results_dir = _ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    _METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved        → {_METRICS_PATH}")

    _HISTORY_PATH.write_text(json.dumps(history, indent=2))
    print(f"History saved        → {_HISTORY_PATH}")

    _save_training_curve(history)
    print(f"Curve saved          → {_CURVE_PATH}")

    class_labels = list(range(n_classes))
    _save_confusion_matrix(all_labels, all_preds, class_labels)
    print(f"Confusion matrix     → {_CM_JSON_PATH}")
    print(f"Confusion matrix png → {_CM_PNG_PATH}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_training_curve(history: dict[str, list[float]]) -> None:
    """Save a two-panel training curve figure (loss + accuracy).

    Parameters
    ----------
    history : dict[str, list[float]]
        Keys: ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss panel
    ax_loss.plot(epochs, history["train_loss"], label="Train", color="#1f77b4")
    ax_loss.plot(epochs, history["val_loss"], label="Validation",
                 color="#ff7f0e", linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Training and Validation Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    # Accuracy panel
    ax_acc.plot(epochs, history["train_acc"], label="Train", color="#1f77b4")
    ax_acc.plot(epochs, history["val_acc"], label="Validation",
                color="#ff7f0e", linestyle="--")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Training and Validation Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle("EEGNet — Motor Imagery Training", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(_CURVE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> None:
    """Compute, save, and plot the confusion matrix for the test set.

    Saves two artefacts:

    * ``results/confusion_matrix.json`` — raw matrix as a nested list with a
      ``labels`` key so the axis orientation is unambiguous.
    * ``results/confusion_matrix.png`` — annotated heatmap figure.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth class indices, shape ``(n_test,)``.
    y_pred : np.ndarray
        Predicted class indices, shape ``(n_test,)``.
    labels : list[int]
        Ordered class labels (e.g. ``[0, 1]``).  Pins the matrix row/column
        order even if a class is absent from the test set.
    """
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required to save the confusion matrix.\n"
            "Install it with:  pip install scikit-learn"
        ) from exc

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # -- JSON ----------------------------------------------------------------
    cm_payload = {
        "labels": labels,
        "matrix": cm.tolist(),
    }
    _CM_JSON_PATH.write_text(json.dumps(cm_payload, indent=2))

    # -- PNG -----------------------------------------------------------------
    n = len(labels)
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")

    tick_labels = [f"Class {lbl}" for lbl in labels]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion Matrix — EEGNet Test Set")

    # Annotate each cell with its count; use white text on dark cells
    thresh = cm.max() / 2.0
    for row in range(n):
        for col in range(n):
            color = "white" if cm[row, col] > thresh else "black"
            ax.text(col, row, str(cm[row, col]),
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    fig.tight_layout()
    fig.savefig(_CM_PNG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EEGNet on the motor imagery dataset."
    )
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial Adam learning rate (default: 1e-3)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
