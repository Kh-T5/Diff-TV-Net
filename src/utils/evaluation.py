import torch
import matplotlib.pyplot as plt
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

        denoised_crop = denoised[..., 2:-2, 2:-2]
        clean_crop = clean[..., 2:-2, 2:-2]

        mse = torch.nn.functional.mse_loss(denoised_crop, clean_crop)
        ssim_val = ssim(denoised_crop, clean_crop, data_range=1.0)

        total_mse += mse.item()
        total_ssim += ssim_val.item()

    return {"mse": total_mse / num_batches, "ssim": total_ssim / num_batches}


def save_debug_plot(noisy, denoised, lam_map, clean, epoch, path):
    """
    Saves a side-by-side comparison to verify spatial adaptivity.
    This is to visualize what the CNN is actively learning during training, better understand what's happening
    """
    img_noisy = noisy[0, 0].cpu().numpy()
    img_denoised = denoised[0, 0].cpu().numpy()
    img_lam = lam_map[0].cpu().numpy()
    img_clean = clean[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img_noisy, cmap="gray")
    axes[0].set_title("Noisy Input")

    axes[1].imshow(img_denoised, cmap="gray")
    axes[1].set_title("Denoised TV)")

    im = axes[2].imshow(img_lam, cmap="hot")
    axes[2].set_title("Learned $\lambda$ Map")
    plt.colorbar(im, ax=axes[2])

    axes[3].imshow(img_clean, cmap="gray")
    axes[3].set_title("Ground Truth")

    for ax in axes:
        ax.axis("off")

    plt.savefig(f"{path}/epoch_{epoch+1}_check.png")
    plt.close()
