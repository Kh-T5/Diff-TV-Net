import torch
from piq import ssim


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluates the model using the loss components defined in the training logic.
    """
    model.eval()

    total_mse = 0.0
    total_ssim = 0.0
    num_batches = len(loader)

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)

        denoised, _ = model(noisy)

        mse = torch.nn.functional.mse_loss(denoised, clean)
        ssim_val = ssim(denoised, clean, data_range=1.0)

        total_mse += mse.item()
        total_ssim += ssim_val.item()

    return {"mse": total_mse / num_batches, "ssim": total_ssim / num_batches}
