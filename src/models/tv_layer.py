import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class DifferentiableTVLayer(nn.Module):
    """
    A Differentiable Convex Optimization Layer for Total Variation Denoising.
    This layer solves the anisotropic or isotropic TV proximal operator using the package "cvxpylayers" to integrate a solver as part of the network
    introduced in the paper: "Differentiable Convex Optimization Layers" from A. Agrawal & al.
    """

    def __init__(self, h, w, reg):
        """
        Inputs:
            h (int): Height of the input image.
            w (int): Width of the input image.
            reg (str): Type of reg to use on gradient, "anisotropic" or "isotropic".
        """

        super().__init__()
        U = cp.Variable((h, w))
        F = cp.Parameter((h, w))
        LAM = cp.Parameter((h, w), nonneg=True)

        if reg == "anisotropic":
            ux = U[1:, :] - U[:-1, :]
            uy = U[:, 1:] - U[:, :-1]
            reg_term = cp.sum(cp.multiply(LAM[1:, :], cp.abs(ux))) + cp.sum(
                cp.multiply(LAM[:, 1:], cp.abs(uy))
            )
        elif reg == "isotropic":
            ux = U[1:, :-1] - U[:-1, :-1]
            uy = U[:-1, 1:] - U[:-1, :-1]

            ux_row = cp.reshape(ux, (1, -1), order="C")
            uy_row = cp.reshape(uy, (1, -1), order="C")

            grads = cp.vstack([ux_row, uy_row])

            grad_norms = cp.norm(grads, p=2, axis=0)

            lam_flat = cp.reshape(LAM[:-1, :-1], (-1,), order="C")
            reg_term = cp.matmul(lam_flat, grad_norms)
        else:
            raise KeyError(f"Regularization '{reg}' not recognized")

        obj = cp.Minimize(0.5 * cp.sum_squares(U - F) + reg_term)
        prob = cp.Problem(obj)

        self.cvx_layer = CvxpyLayer(prob, parameters=[F, LAM], variables=[U])

    def forward(self, f, lam):
        """
        Forward pass in the CVXPY layer using SCS solver.
        Inputs:
            f (torch.Tensor): Noisy image tensor of shape (Batch, H, W).
            lam (torch.Tensor): Regularization map of shape (Batch, H, W).

        Returns:
            torch.Tensor: Optimal solution U* (denoised image) of shape (Batch, H, W).
        """
        orig_device = f.device
        f_cpu = f.to("cpu").to(torch.float32)
        lam_cpu = lam.to("cpu").to(torch.float32)

        solver_info = {"solve_method": cp.SCS, "max_iters": 500, "eps": 5e-3}
        result = self.cvx_layer(f_cpu, lam_cpu, solver_args=solver_info)[0].to(
            torch.float32
        )

        return result.to(orig_device)
