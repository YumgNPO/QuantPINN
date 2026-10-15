"""
Physics-Informed Neural Network (PINN) for the 2D Wave Equation in PyTorch.
Supports both soft boundary/initial penalty losses (v1 baseline) and
hard-constrained trial ansatz parameterization (v2 fix).
"""

from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn


class WavePINN2D(nn.Module):
    def __init__(
        self,
        hidden_layers: int = 4,
        hidden_units: int = 64,
        c: float = 1.0,
        use_hard_ansatz: bool = True,
    ):
        super().__init__()
        self.c = c
        self.use_hard_ansatz = use_hard_ansatz

        layers = [nn.Linear(3, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def exact_ic_u0(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Initial displacement u(x, y, 0)."""
        return (
            torch.sin(np.pi * x) * torch.sin(np.pi * y)
            + 0.5 * torch.sin(2.0 * np.pi * x) * torch.sin(np.pi * y)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates u_PINN(x, y, t).
        When use_hard_ansatz=True, satisfies u(x,y,0) = u0(x,y), u_t(x,y,0) = 0,
        and Dirichlet boundaries u(x_b, y_b, t) = 0 identically by construction:
            u(x,y,t) = u0(x,y) + t^2 * x(1-x)y(1-y) * N_theta(x,y,t)
        """
        # Map [0, 1] inputs to [-1, 1] for balanced Tanh gradients
        inp = 2.0 * torch.cat([x, y, t], dim=1) - 1.0
        raw_out = self.net(inp)

        if not self.use_hard_ansatz:
            return raw_out

        u0 = self.exact_ic_u0(x, y)
        spatial_bubble = x * (1.0 - x) * y * (1.0 - y)
        return u0 + (t**2) * spatial_bubble * raw_out

    def pde_residual(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the 2D wave PDE residual: R = u_tt - c^2 * (u_xx + u_yy)
        via exact PyTorch automatic differentiation.
        """
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        u = self.forward(x, y, t)
        ones = torch.ones_like(u)

        u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=ones, create_graph=True)[0]

        residual = u_tt - (self.c**2) * (u_xx + u_yy)
        return residual

    def compute_loss(
        self,
        collocation_pts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ic_pts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        bc_pts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        w_ic: float = 100.0,
        w_bc: float = 100.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes composite PINN loss. If use_hard_ansatz=True, IC and BC losses are
        identically zero and only the interior PDE residual is needed.
        """
        x_c, y_c, t_c = collocation_pts
        res_pde = self.pde_residual(x_c, y_c, t_c)
        loss_pde = torch.mean(res_pde**2)

        if self.use_hard_ansatz or ic_pts is None or bc_pts is None:
            metrics = {
                "loss_total": float(loss_pde.item()),
                "loss_pde": float(loss_pde.item()),
                "loss_ic": 0.0,
                "loss_bc": 0.0,
            }
            return loss_pde, metrics

        # Soft IC/BC path (for v1 ablation comparisons)
        x_i, y_i, t_i = ic_pts
        t_i.requires_grad_(True)
        u_pred_ic = self.forward(x_i, y_i, t_i)
        u_t_ic = torch.autograd.grad(
            u_pred_ic, t_i, grad_outputs=torch.ones_like(u_pred_ic), create_graph=True
        )[0]
        loss_ic = torch.mean((u_pred_ic - self.exact_ic_u0(x_i, y_i)) ** 2) + torch.mean(
            u_t_ic**2
        )

        x_b, y_b, t_b = bc_pts
        u_pred_bc = self.forward(x_b, y_b, t_b)
        loss_bc = torch.mean(u_pred_bc**2)

        total_loss = loss_pde + w_ic * loss_ic + w_bc * loss_bc
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_pde": float(loss_pde.item()),
            "loss_ic": float(loss_ic.item()),
            "loss_bc": float(loss_bc.item()),
        }
        return total_loss, metrics