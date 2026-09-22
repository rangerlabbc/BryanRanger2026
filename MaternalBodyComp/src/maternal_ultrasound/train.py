# =============================================================================
# train.py
# Training loop and evaluation metrics for maternal body composition UNet.
#
# USAGE:
#   from maternal_ultrasound.train import train, evaluate
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Loss functions
# =============================================================================

class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error loss."""
    def forward(self, pred, target):
        return torch.mean(torch.abs((target - pred) / (target + 1e-8))) * 100


class CustomLoss(nn.Module):
    """
    Weighted combination of MSE + L1 + penalty for negative predictions.
    Used as the default for UNet (matches original notebook behaviour).
    """
    def __init__(self, weight_mse=0.6, weight_l1=0.4):
        super().__init__()
        self.weight_mse = weight_mse
        self.weight_l1  = weight_l1

    def forward(self, pred, target):
        mse     = F.mse_loss(pred, target)
        l1      = F.l1_loss(pred, target)
        penalty = torch.mean(F.relu(-pred))   # penalise negative predictions
        return self.weight_mse * mse + self.weight_l1 * l1 + penalty


def _get_criterion(name: str):
    """Return the loss function for a given name string."""
    options = {
        "MSE":    nn.MSELoss(),
        "MAE":    nn.L1Loss(),
        "MAPE":   MAPELoss(),
        "Custom": CustomLoss(),
    }
    if name not in options:
        raise ValueError(f"Unknown criterion '{name}'. Choose from {list(options)}")
    return options[name]


# =============================================================================
# Training loop
# =============================================================================

def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 90,
    criterion: str = "Custom",
    lr: float = 0.001,
    adaptive_lr: bool = True,
    scheduler_gamma: float = 0.95,
) -> tuple:
    """
    Train the UNet model.

    Args:
        model         : UNet instance (already moved to device)
        train_loader  : DataLoader for training patients
        val_loader    : DataLoader for validation patients
        device        : torch.device ("cpu" or "cuda" or "mps")
        epochs        : number of epochs
        criterion     : loss function name ("MSE", "MAE", "MAPE", "Custom")
        lr            : initial learning rate
        adaptive_lr   : decay lr each epoch with ExponentialLR
        scheduler_gamma: gamma for ExponentialLR

    Returns:
        (train_loss_history, val_loss_history) — one float per epoch
    """
    loss_fn   = _get_criterion(criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
        if adaptive_lr else None
    )

    train_history = []
    val_history   = []

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images  = [img.to(device) for img in batch[0]]
            y_true  = batch[1].to(device).float()

            optimizer.zero_grad()
            y_pred = model(images)

            # Ensure matching shapes for loss
            y_pred = y_pred.view_as(y_true)
            loss   = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  "
                      f"Batch {batch_idx}  Loss: {loss.item():.4f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = [img.to(device) for img in batch[0]]
                y_true = batch[1].to(device).float()
                y_pred = model(images)
                y_pred = y_pred.view_as(y_true)
                val_loss += loss_fn(y_pred, y_true).item()

        train_epoch_loss = train_loss / len(train_loader)
        val_epoch_loss   = val_loss   / len(val_loader)

        print("=" * 70)
        print(f"Epoch {epoch+1} complete — "
              f"Train loss: {train_epoch_loss:.4f}  "
              f"Val loss: {val_epoch_loss:.4f}")
        print("=" * 70)

        train_history.append(train_epoch_loss)
        val_history.append(val_epoch_loss)

        if scheduler:
            scheduler.step()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return train_history, val_history


# =============================================================================
# Evaluation metrics
# =============================================================================

def evaluate(model, loader, device) -> dict:
    """
    Run model on all batches in loader and return evaluation metrics.

    Returns dict with keys: MAE, MSE, RMSE, MAPE,
    plus lists true_labels and predictions.
    """
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            images = [img.to(device) for img in batch[0]]
            y_true = batch[1].to(device).float()
            y_pred = model(images).view_as(y_true)
            all_true.append(y_true.cpu())
            all_pred.append(y_pred.cpu())

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)

    mae  = torch.mean(torch.abs(y_true - y_pred)).item()
    mse  = torch.mean((y_true - y_pred) ** 2).item()
    rmse = math.sqrt(mse)
    mape = (torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8))) * 100).item()

    return {
        "MAE":          mae,
        "MSE":          mse,
        "RMSE":         rmse,
        "MAPE":         mape,
        "true_labels":  y_true.tolist(),
        "predictions":  y_pred.tolist(),
    }
