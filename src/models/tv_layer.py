import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class DifferentiableTVLayer(nn.Module):
    """
    A Differentiable Convex Optimization Layer for Total Variation Denoising.
    This layer solves the anisotropic TV proximal operator using the package "cvxpylayers"
    introduced in the paper: "Differentiable Convex Optimization Layers" from A. Agrawal & al.
    Using the Implicit Function Theorem on the KKT conditions, it allows backpropagation from the output image 'u' to the parameters 'f' and 'lambda'.
    """

    def __init__(self, h, w):
        """
        Inputs:
            h (int): Height of the input image.
            w (int): Width of the input image.
        """

        super().__init__()
        U = cp.Variable((h, w))
        F = cp.Parameter((h, w))  # Noisy input
        LAM = cp.Parameter((h, w), nonneg=True)

        # Finite difference operators (Anisotropic TV)
        ux = U[1:, :] - U[:-1, :]
        uy = U[:, 1:] - U[:, :-1]

        reg_term = cp.sum(cp.multiply(LAM[1:, :], cp.abs(ux))) + cp.sum(
            cp.multiply(LAM[:, 1:], cp.abs(uy))
        )

        obj = cp.Minimize(0.5 * cp.sum_squares(U - F) + reg_term)
        prob = cp.Problem(obj)

        self.cvx_layer = CvxpyLayer(prob, parameters=[F, LAM], variables=[U])

    def forward(self, f, lam):
        """
        Inputs:
            f (torch.Tensor): Noisy image tensor of shape (Batch, H, W).
            lam (torch.Tensor): Regularization map of shape (Batch, H, W).

        Returns:
            torch.Tensor: Optimal solution U* (denoised image) of shape (Batch, H, W).
        """

        return self.cvx_layer(f, lam)[0]
