import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class GaussianDenoisingDataset(Dataset):
    """
    Dataset for Supervised Denoising.
    Loads clean images, converts to grayscale, applies additive white Gaussian noise.
    """

    def __init__(self, root_dir, patch_size=64, sigma=25):
        """
        Inputs:
            - root_dir (str): Directory with images.
            - patch_size (int): Size of the square crop for the CVX layer.
            - sigma (float): Standard deviation of the Gaussian noise.
        """

        self.root_dir = root_dir
        self.image_filenames = [f for f in os.listdir(root_dir) if f.endswith((".png"))]
        self.patch_size = patch_size
        self.sigma = sigma / 255.0
        self.base_transform = transforms.Compose(
            [
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Reads the saved image file given the index and returns two versions:
        A clean patch and a noisy patch with additive white noise.
        """

        img_path = os.path.join(self.root_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("L")

        clean_patch = self.base_transform(image).to(torch.float32)
        noise = torch.randn_like(clean_patch) * self.sigma
        noisy_patch = clean_patch + noise
        noisy_patch = torch.clamp(noisy_patch, 0, 1)

        return noisy_patch.to(torch.float32), clean_patch.to(torch.float32)
