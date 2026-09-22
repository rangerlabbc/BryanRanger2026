# =============================================================================
# run_training.py
# Entry point for training the maternal UNet body composition model.
#
# USAGE (from the MaternalBodyComp/ root with venv active):
#   python scripts/run_training.py
#
# All experiment settings are in configs/training.yaml
# =============================================================================

import os
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
from torch.utils.data import DataLoader, random_split

from maternal_ultrasound.dataset import MaternalDataset
from maternal_ultrasound.model import UNet
from maternal_ultrasound.train import train, evaluate


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Device selection
# Prefers CUDA (server GPU) → MPS (Apple Silicon) → CPU
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU — training will be slow. Consider running on server.")
    return device


# =============================================================================
# Main
# =============================================================================

def main():
    # Load config
    config_path = Path("configs/training.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = get_device()

    # ------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------
    # Discover all patient IDs from the processed data folder
    data_root = Path(cfg["data"]["processed_dir"])
    all_patients = sorted([
        p.name for p in data_root.iterdir()
        if p.is_dir() and not p.name.startswith('.')
    ])
    print(f"Found {len(all_patients)} patient folders in {data_root}")

    full_dataset = MaternalDataset(
        root_dir      = cfg["data"]["processed_dir"],
        labels_csv    = cfg["data"]["labels_csv"],
        patient_ids   = all_patients,
        region_combo  = cfg["data"]["region_combo"],
        num_images    = cfg["data"]["num_images"],
        output        = cfg["data"]["output"],
        crop          = cfg["data"].get("crop", [1.0, 1.0, 1.0, 1.0]),
        speckle       = cfg["data"].get("speckle", False),
        despeckle     = cfg["data"].get("despeckle", False),
        augment       = cfg["data"].get("augment", 0),
    )

    print(f"Dataset size after matching with CSV: {len(full_dataset)} patients")

    # Train / val split (patient-level, not image-level)
    train_frac  = cfg["data"].get("train_fraction", 0.9)
    train_size  = int(train_frac * len(full_dataset))
    val_size    = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    print(f"Train: {train_size} patients  |  Val: {val_size} patients")

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              collate_fn=_collate)
    val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False,
                              collate_fn=_collate)

    # ------------------------------------------------------------------
    # Build model
    # Number of images expected per patient =
    #   num_images × number of regions × number of timepoints loaded
    # For a first run: 2 images × 3 regions = 6 → n_images=6
    # Update this if you change num_images or region_combo.
    # ------------------------------------------------------------------
    n_images = cfg["model"].get("n_images", 6)
    model = UNet(
        n_channels = cfg["model"].get("input_channels", 3),
        n_classes  = cfg["model"].get("output_dim", 1),
        n_images   = n_images,
        bilinear   = cfg["model"].get("bilinear", True),
    ).to(device)

    print(f"\nModel: UNet  |  n_images={n_images}  |  device={device}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    train_history, val_history = train(
        model         = model,
        train_loader  = train_loader,
        val_loader    = val_loader,
        device        = device,
        epochs        = cfg["training"]["epochs"],
        criterion     = cfg["training"]["criterion"],
        lr            = cfg["training"]["learning_rate"],
        adaptive_lr   = cfg["training"].get("adaptive_lr", True),
        scheduler_gamma= cfg["training"].get("scheduler_gamma", 0.95),
    )

    # ------------------------------------------------------------------
    # Plot loss curves
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label="Train Loss")
    plt.plot(val_history,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"UNet — {cfg['data']['output']} — {cfg['data']['region_combo']}")
    plt.legend()
    plt.grid(True)

    plots_dir = Path(cfg["artifacts"].get("results_dir", "artifacts/results"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"loss_{cfg['data']['output']}_{cfg['data']['region_combo']}_{date.today()}.png"
    plt.savefig(plot_path)
    print(f"\nLoss curve saved to {plot_path}")

    # ------------------------------------------------------------------
    # Evaluate on validation set
    # ------------------------------------------------------------------
    metrics = evaluate(model, val_loader, device)
    print("\n=== Validation Metrics ===")
    for k, v in metrics.items():
        if k not in ("true_labels", "predictions"):
            print(f"  {k}: {v:.4f}")

    print("\nTrue labels vs predictions:")
    for true, pred in zip(metrics["true_labels"], metrics["predictions"]):
        print(f"  True: {true:.4f}  Pred: {pred:.4f}")

    # ------------------------------------------------------------------
    # Save model checkpoint
    # ------------------------------------------------------------------
    checkpoint_dir = Path(cfg["artifacts"].get("checkpoint_dir", "artifacts/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_name = (
        f"UNet_{cfg['training']['criterion']}_"
        f"{cfg['data']['region_combo']}_{cfg['data']['output']}_"
        f"{cfg['training']['epochs']}epochs_{date.today()}.pt"
    )
    checkpoint_path = checkpoint_dir / run_name
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel saved to {checkpoint_path}")


# =============================================================================
# Collate function
# Each patient returns a list of tensors (variable length).
# This custom collate keeps them as lists rather than stacking,
# which is what the model forward() expects.
# =============================================================================

def _collate(batch):
    images  = [item[0] for item in batch]   # list of lists of tensors
    labels  = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    weights = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    lengths = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    # Flatten to a single list of tensors (batch_size=1 so this is fine)
    images_flat = images[0] if len(images) == 1 else [img for sublist in images for img in sublist]
    return images_flat, labels, weights, lengths


if __name__ == "__main__":
    main()
