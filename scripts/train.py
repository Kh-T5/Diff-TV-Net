import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os

from src.models.nn_hybrid import TVDenoisingNet
from src.utils.trainer import train_one_epoch
from src.utils.evaluation import evaluate
from src.data.dataset import GaussianDenoisingDataset
from src.config import (
    data_dir,
    patch_size,
    gaussian_std,
    device_name,
    results_path,
    model_path,
)


def main(args):
    """
    Runs the training of our model on BDS500 data.
    """
    histories = {"loss": [], "mse": [], "ssim": []}
    device = torch.device(device_name if torch.mps.is_available() else "cpu")
    print(f"Device used: {device}\n")

    model = TVDenoisingNet(img_size=(patch_size, patch_size))
    model = model.to(torch.float32)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

    training_dir = os.path.join(data_dir, "train")
    train_dataset = GaussianDenoisingDataset(training_dir, patch_size, gaussian_std)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print("Train Loader initialized.")

    print(f"Starting Training | SSIM: {args.use_ssim} | Alpha: {args.alpha}")

    for epoch in range(args.epochs):
        epoch_loss = train_one_epoch(
            model, loader, optimizer, device, args.alpha, args.use_ssim
        )
        histories["loss"].append(epoch_loss)
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            validation_dir = os.path.join(data_dir, "val")
            val_dataset = GaussianDenoisingDataset(
                validation_dir, patch_size, gaussian_std
            )
            val_loader = DataLoader(
                val_dataset, batch_size=int(3 * args.batch_size), shuffle=False
            )
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"Validation MSE: {val_metrics['mse']:.2f} | SSIM: {val_metrics['ssim']:.4f}"
            )
            histories["mse"].append(val_metrics["mse"])
            histories["ssim"].append(val_metrics["ssim"])

    np.savez(results_path, **histories, config=vars(args))
    torch.save(model.state_dict(), model_path)
    print(f"Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TV-Opti-Net Training")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--alpha", type=float, default=0.8, help="Weight for MSE in dual-loss"
    )
    parser.add_argument(
        "--use_ssim",
        default=True,
        action="store_true",
        help="Toggle SSIM loss component",
    )

    args = parser.parse_args()
    main(args)
