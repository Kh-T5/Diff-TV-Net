import torch.nn as nn
import torch
from piq import ssim
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: str,
    alpha: float,
    use_ssim: bool,
) -> float:
    """
    Runs one epoch for the model.
    Integrates gradient clipping for stability.

    Inputs:
        - model: model to be trained, nn.Module
        - loader: dataloader from torch.utils.data
        - optimizer: torch optimizer, usually ADAM
        - device: device for training, can be "cpu", "cuda", "mps"
        - alpha: float in [0, 1] refers to hte weight attributed to each part of the dual loss
        - use_ssim: bool, indicates if only MSE is used in loss or if we use dual loss with SSIM

    Returns:
        mean loss over epoch
    """
    model.train()
    total_loss = 0

    for batch_idx, (noisy, clean) in enumerate(loader):
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()

        denoised, _ = model(noisy)

        mse_loss = nn.functional.mse_loss(denoised, clean)

        if use_ssim:
            ssim_val = ssim(denoised, clean, data_range=1.0)
            loss = alpha * mse_loss + (1 - alpha) * (1 - ssim_val)
        else:
            loss = mse_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)
