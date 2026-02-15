import torch.nn.functional as F
import torch.nn as nn
from src.models.tv_layer import DifferentiableTVLayer
import torch


class TVDenoisingNet(nn.Module):
    """
    Hybrid Neural Network for Denoising.
    Architecture:
        CNN Weight Predictor: Analyzes noise features to produce a spatial map lambda.
        Softplus Activation: Ensures mathematical validity (lambda > 0) for conexity guarantee.
        Differentiable TV Layer: Solves the optimization problem to produce the denoised image.
    """

    def __init__(self, img_size: tuple[int, int] = (64, 64)):
        """
        Initalize the neural network.
        Inputs:
            img_size (tuple): Dimensions (H, W) for the optimization problem.
        """
        super().__init__()
        self.h, self.w = img_size

        self.weight_predictor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Softplus(),
        )

        self.tv_layer = DifferentiableTVLayer(self.h, self.w)

    def forward(self, x):
        """
        Performs forward pass.
        Inputs:
            x (torch.Tensor): Noisy input image, shape (Batch, 1, H, W).

        Returns:
            denoised (torch.Tensor): Denoised output, shape (Batch, 1, H, W).
            lam_map (torch.Tensor): The learned regularization map, shape (Batch, H, W).
        """
        f = x.squeeze(1)

        lam_map = self.weight_predictor(x).squeeze(1)

        denoised = self.tv_layer(f, lam_map)

        return denoised.unsqueeze(1), lam_map

    def to(self, *args, **kwargs):
        """
        Encountered hardware problems, this is to ensure that optimization layer is not put on
        Apple Sillicon GPU which is not supported, therefore only "classic" layers like the CNN
        at the start is switched to the gpu.
        """
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None and device.type == "mps":
            for name, module in self.named_children():
                if name != "tv_layer":
                    module.to(device)
                else:
                    module.to("cpu")
            return self
        return super().to(*args, **kwargs)
