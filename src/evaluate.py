"""
Evaluation metrics (Relative L2, L_infinity, Final-Slice Max Error, RMSE) and
publication-quality figure generation for QuantPINN vs. Finite-Difference Reference.
"""

import os
from typing import Dict, List
import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_benchmark_metrics(u_pinn: np.ndarray, u_fd: np.ndarray) -> Dict[str, float]:
    """
    Computes grid-aligned Relative L2 error, global Max Absolute Error (L_inf),
    final-slice Max Absolute Error at t = t_max, and RMSE.
    """
    assert (
        u_pinn.shape == u_fd.shape
    ), f"Grid shape mismatch: PINN {u_pinn.shape} vs FD {u_fd.shape}"

    diff = u_pinn - u_fd
    rel_l2 = np.linalg.norm(diff.ravel(), ord=2) / np.linalg.norm(u_fd.ravel(), ord=2)
    max_abs = np.max(np.abs(diff))
    final_slice_max_abs = np.max(np.abs(diff[-1]))
    rmse = np.sqrt(np.mean(diff**2))

    return {
        "relative_l2_error": float(rel_l2),
        "relative_l2_percent": float(rel_l2 * 100.0),
        "max_abs_error": float(max_abs),
        "final_slice_max_abs_error": float(final_slice_max_abs),
        "rmse": float(rmse),
    }


@torch.no_grad()
def predict_on_fd_grid(
    model: torch.nn.Module,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    device: torch.device,
    batch_size: int = 32768,
) -> np.ndarray:
    """
    Evaluates the trained PINN across the exact (nt, nx, ny) Finite-Difference grid
    using mini-batches to avoid GPU memory spikes.
    """
    model.eval()
    nt, nx, ny = len(t_grid), len(x_grid), len(y_grid)
    T, X, Y = np.meshgrid(t_grid, x_grid, y_grid, indexing="ij")

    pts_x = torch.tensor(X.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
    pts_y = torch.tensor(Y.ravel(), dtype=torch.float32, device=device).unsqueeze(1)
    pts_t = torch.tensor(T.ravel(), dtype=torch.float32, device=device).unsqueeze(1)

    preds = []
    n_total = pts_x.shape[0]
    for i in range(0, n_total, batch_size):
        u_batch = model(
            pts_x[i : i + batch_size],
            pts_y[i : i + batch_size],
            pts_t[i : i + batch_size],
        )
        preds.append(u_batch.cpu().numpy())

    u_pinn = np.vstack(preds).reshape(nt, nx, ny)
    return u_pinn


def generate_all_figures(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    u_fd: np.ndarray,
    u_pinn: np.ndarray,
    history: Dict[str, List[float]],
    output_dir: str = "figures",
) -> None:
    """
    Generates and saves the three required diagnostic figures into `figures/`.
    """
    os.makedirs(output_dir, exist_ok=True)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    # -------------------------------------------------------------------------
    # Figure 1: FD Reference vs. PINN Reconstruction at 3 Time Snapshots
    # -------------------------------------------------------------------------
    snap_indices = [0, len(t_grid) // 2, len(t_grid) - 1]
    fig1, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    vmin, vmax = np.min(u_fd), np.max(u_fd)
    for col, idx in enumerate(snap_indices):
        t_val = t_grid[idx]

        im0 = axes[0, col].pcolormesh(
            X, Y, u_fd[idx], shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        axes[0, col].set_title(
            f"FD Reference (t = {t_val:.2f} s)", fontsize=11, fontweight="bold"
        )
        axes[0, col].set_xlabel("x")
        axes[0, col].set_ylabel("y")
        fig1.colorbar(im0, ax=axes[0, col], shrink=0.85)

        im1 = axes[1, col].pcolormesh(
            X, Y, u_pinn[idx], shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax
        )
        axes[1, col].set_title(
            f"QuantPINN Prediction (t = {t_val:.2f} s)", fontsize=11, fontweight="bold"
        )
        axes[1, col].set_xlabel("x")
        axes[1, col].set_ylabel("y")
        fig1.colorbar(im1, ax=axes[1, col], shrink=0.85)

    fig1.suptitle(
        "2D Wave Equation: Finite-Difference Ground Truth vs. QuantPINN",
        fontsize=14,
        fontweight="bold",
    )
    fig1.savefig(os.path.join(output_dir, "fig1_solution_comparison.png"), dpi=300)
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Figure 2: Pointwise Absolute Error Maps |u_PINN - u_FD|
    # -------------------------------------------------------------------------
    abs_err = np.abs(u_pinn - u_fd)
    mid_y_idx = len(y_grid) // 2
    T_xt, X_xt = np.meshgrid(t_grid, x_grid, indexing="ij")

    fig2, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )

    im_err_xy = ax_left.pcolormesh(X, Y, abs_err[-1], shading="auto", cmap="magma")
    ax_left.set_title(
        f"Spatial Pointwise Error |u_PINN - u_FD| at t = {t_grid[-1]:.2f} s",
        fontsize=11,
        fontweight="bold",
    )
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("y")
    fig2.colorbar(im_err_xy, ax=ax_left, label="Absolute Error")

    im_err_xt = ax_right.pcolormesh(
        T_xt, X_xt, abs_err[:, :, mid_y_idx], shading="auto", cmap="magma"
    )
    ax_right.set_title(
        f"Space-Time Error Evolution at Cross-Section y = {y_grid[mid_y_idx]:.2f}",
        fontsize=11,
        fontweight="bold",
    )
    ax_right.set_xlabel("Time t (s)")
    ax_right.set_ylabel("x")
    fig2.colorbar(im_err_xt, ax=ax_right, label="Absolute Error")

    fig2.suptitle(
        "Pointwise Absolute Error Diagnostics Across Space and Time",
        fontsize=13,
        fontweight="bold",
    )
    fig2.savefig(os.path.join(output_dir, "fig2_pointwise_error_map.png"), dpi=300)
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # Figure 3: Training Loss Landscape & Validation Relative L2 Convergence
    # -------------------------------------------------------------------------
    fig3, (ax_loss, ax_l2) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )

    epochs_loss = np.arange(1, len(history["loss_pde"]) + 1)
    ax_loss.semilogy(
        epochs_loss,
        history["loss_pde"],
        label="PDE Residual Loss",
        color="#1f77b4",
        lw=1.8,
    )
    if "adam_epochs" in history and history["adam_epochs"] < len(epochs_loss):
        ax_loss.axvline(
            history["adam_epochs"],
            color="#d62728",
            linestyle="--",
            lw=1.4,
            label="Switch: Adam -> L-BFGS",
        )
    ax_loss.set_title("PDE Residual Loss Convergence", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Optimization Step")
    ax_loss.set_ylabel("Mean Squared PDE Residual (log scale)")
    ax_loss.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_loss.legend()

    ax_l2.semilogy(
        history["eval_steps"],
        history["rel_l2_history"],
        marker="o",
        markersize=4,
        color="#2ca02c",
        lw=1.8,
        label="Validation Relative L2 Error (vs. FD)",
    )
    if "adam_epochs" in history and history["adam_epochs"] < len(epochs_loss):
        ax_l2.axvline(
            history["adam_epochs"],
            color="#d62728",
            linestyle="--",
            lw=1.4,
            label="Switch: Adam -> L-BFGS",
        )
    ax_l2.set_title(
        "True Validation Relative L2 Error vs. FD Reference",
        fontsize=11,
        fontweight="bold",
    )
    ax_l2.set_xlabel("Optimization Step")
    ax_l2.set_ylabel("Relative L2 Error ||u_PINN - u_FD||_2 / ||u_FD||_2")
    ax_l2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_l2.legend()

    fig3.suptitle(
        "QuantPINN Optimization Trajectory & Finite-Difference Validation Audit",
        fontsize=13,
        fontweight="bold",
    )
    fig3.savefig(os.path.join(output_dir, "fig3_training_convergence.png"), dpi=300)
    plt.close(fig3)