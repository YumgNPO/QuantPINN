"""
Single entry-point script to run the Finite-Difference reference solver,
train QuantPINN on GPU/CPU, compute grid-aligned L2 errors against both FD
and the exact analytical solution, and save all figures.

Usage:
    python scripts/train_and_benchmark.py
    python scripts/train_and_benchmark.py --fd-only
    python scripts/train_and_benchmark.py --naive-start
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fd_reference import solve_fd_wave_2d
from src.model_pinn import WavePINN2D
from src.evaluate import (
    compute_benchmark_metrics,
    predict_on_fd_grid,
    generate_all_figures,
)


def sample_collocation_points(
    n_pts: int, t_max: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.rand((n_pts, 1), dtype=torch.float32, device=device)
    y = torch.rand((n_pts, 1), dtype=torch.float32, device=device)
    t = torch.rand((n_pts, 1), dtype=torch.float32, device=device) * t_max
    return x, y, t


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantPINN 2D Wave Benchmark")
    parser.add_argument(
        "--naive-start",
        action="store_true",
        help="Use first-order naive initialization u^1 = u^0 + dt*v0 in FD reference.",
    )
    parser.add_argument(
        "--fd-only",
        action="store_true",
        help="Run only the Finite-Difference Taylor vs. Naive-Start ablation and exit.",
    )
    args, _ = parser.parse_known_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    print(f"[QuantPINN] Running on device: {device} ({device_name})")

    c_wave = 1.0
    t_max = 0.5
    nx, ny, nt = 81, 81, 251
    n_collocation = 12000
    adam_epochs = 2500
    lbfgs_max_iter = 500
    eval_interval = 250

    use_taylor = not args.naive_start
    print(
        f"[QuantPINN] Solving 2D Finite-Difference Reference ({nx}x{ny}x{nt}, "
        f"use_taylor_first_step={use_taylor})..."
    )
    x_grid, y_grid, t_grid, u_fd, u_exact, fd_meta = solve_fd_wave_2d(
        nx=nx,
        ny=ny,
        nt=nt,
        t_max=t_max,
        c=c_wave,
        use_taylor_first_step=use_taylor,
    )
    print(
        f"  -> FD Solve Time: {fd_meta['fd_wall_clock_ms']:.2f} ms | "
        f"CFL = {fd_meta['cfl']:.4f}"
    )
    print(
        f"  -> FD (Taylor n=1) vs. Exact L2 (Full): {fd_meta['fd_vs_exact_rel_l2_full']:.3e} | "
        f"(Final): {fd_meta['fd_vs_exact_rel_l2_final']:.3e}"
    )
    print(
        f"  -> FD (Naive n=1)  vs. Exact L2 (Full): {fd_meta['fd_naive_vs_exact_rel_l2_full']:.3e} | "
        f"Improvement Factor: {fd_meta['fd_taylor_improvement_factor']:.1f}x"
    )

    if args.fd_only:
        return

    model = WavePINN2D(
        hidden_layers=4, hidden_units=64, c=c_wave, use_hard_ansatz=True
    ).to(device)

    history = {
        "loss_pde": [],
        "eval_steps": [],
        "rel_l2_history": [],
        "adam_epochs": adam_epochs,
    }

    start_time = time.perf_counter()

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_adam, T_max=adam_epochs, eta_min=1e-4
    )

    colloc_pts = sample_collocation_points(n_collocation, t_max, device)

    print(f"[QuantPINN] Stage 1: Training with Adam for {adam_epochs} epochs...")
    for epoch in range(1, adam_epochs + 1):
        model.train()
        if epoch % 500 == 0:
            colloc_pts = sample_collocation_points(n_collocation, t_max, device)

        optimizer_adam.zero_grad()
        loss, metrics = model.compute_loss(colloc_pts)
        loss.backward()
        optimizer_adam.step()
        scheduler.step()

        history["loss_pde"].append(metrics["loss_pde"])

        if epoch == 1 or epoch % eval_interval == 0:
            u_pred = predict_on_fd_grid(model, x_grid, y_grid, t_grid, device)
            eval_metrics = compute_benchmark_metrics(u_pred, u_fd)
            history["eval_steps"].append(epoch)
            history["rel_l2_history"].append(eval_metrics["relative_l2_error"])
            print(
                f"  [Adam {epoch:4d}/{adam_epochs}] "
                f"PDE Loss: {metrics['loss_pde']:.3e} | "
                f"Rel L2 vs FD: {eval_metrics['relative_l2_error']:.3e} "
                f"({eval_metrics['relative_l2_percent']:.3f}%)"
            )

    print(
        f"[QuantPINN] Stage 2: Refining with L-BFGS (max {lbfgs_max_iter} iterations)..."
    )
    colloc_pts_lbfgs = sample_collocation_points(n_collocation, t_max, device)
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=lbfgs_max_iter,
        max_eval=int(lbfgs_max_iter * 1.25),
        history_size=50,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    step_counter = [adam_epochs]

    def closure():
        optimizer_lbfgs.zero_grad()
        loss, metrics = model.compute_loss(colloc_pts_lbfgs)
        loss.backward()

        step_counter[0] += 1
        history["loss_pde"].append(metrics["loss_pde"])

        if (step_counter[0] - adam_epochs) % 100 == 0:
            u_pred = predict_on_fd_grid(model, x_grid, y_grid, t_grid, device)
            eval_metrics = compute_benchmark_metrics(u_pred, u_fd)
            history["eval_steps"].append(step_counter[0])
            history["rel_l2_history"].append(eval_metrics["relative_l2_error"])
            print(
                f"  [L-BFGS Eval {step_counter[0] - adam_epochs:4d}] "
                f"PDE Loss: {metrics['loss_pde']:.3e} | "
                f"Rel L2 vs FD: {eval_metrics['relative_l2_error']:.3e} "
                f"({eval_metrics['relative_l2_percent']:.3f}%)"
            )
        return loss

    model.train()
    optimizer_lbfgs.step(closure)

    wall_time = time.perf_counter() - start_time

    u_pinn_final = predict_on_fd_grid(model, x_grid, y_grid, t_grid, device)
    final_fd_metrics = compute_benchmark_metrics(u_pinn_final, u_fd)
    final_exact_metrics = compute_benchmark_metrics(u_pinn_final, u_exact)

    history["eval_steps"].append(step_counter[0])
    history["rel_l2_history"].append(final_fd_metrics["relative_l2_error"])

    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    generate_all_figures(
        x_grid, y_grid, t_grid, u_fd, u_pinn_final, history, output_dir=figures_dir
    )

    summary = {
        **final_fd_metrics,
        "pinn_vs_exact_rel_l2": final_exact_metrics["relative_l2_error"],
        "pinn_vs_exact_max_abs": final_exact_metrics["max_abs_error"],
        **fd_meta,
        "final_pde_loss": history["loss_pde"][-1],
        "wall_clock_seconds": round(wall_time, 2),
        "device": device_name,
        "adam_epochs": adam_epochs,
        "lbfgs_closure_evals": step_counter[0] - adam_epochs,
        "total_steps": step_counter[0],
    }
    with open(os.path.join(figures_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"- PINN vs. FD Relative L2:      {final_fd_metrics['relative_l2_error']:.3e} ({final_fd_metrics['relative_l2_percent']:.3f}%)")
    print(f"- PINN vs. Exact Relative L2:   {final_exact_metrics['relative_l2_error']:.3e} ({final_exact_metrics['relative_l2_percent']:.3f}%)")
    print(f"- Global Max Pointwise Error:   {final_fd_metrics['max_abs_error']:.3e}")
    print(f"- Final-Slice Max Error (t=0.5):{final_fd_metrics['final_slice_max_abs_error']:.3e}")
    print(f"- Root Mean Squared Error:      {final_fd_metrics['rmse']:.3e}")
    print(f"- Final PDE Residual Loss:      {history['loss_pde'][-1]:.3e}")
    print(f"- FD Reference CFL Number:      {fd_meta['cfl']:.4f}")
    print(f"- FD Wall-Clock Time (CPU):     {fd_meta['fd_wall_clock_ms']:.2f} ms")
    print(f"- FD vs. Exact L2 (Taylor Full):{fd_meta['fd_vs_exact_rel_l2_full']:.3e}")
    print(f"- FD vs. Exact L2 (Naive Full): {fd_meta['fd_naive_vs_exact_rel_l2_full']:.3e} ({fd_meta['fd_taylor_improvement_factor']:.1f}x worse)")
    print(f"- FD vs. Exact L2 (Final Slice):{fd_meta['fd_vs_exact_rel_l2_final']:.3e}")
    print(f"- Total Evaluations:            {step_counter[0]} ({adam_epochs} Adam + {step_counter[0] - adam_epochs} L-BFGS evals)")
    print(f"- PINN Wall-Clock Time (GPU):   {wall_time:.1f} s on {device_name}")
    print("=" * 72)


if __name__ == "__main__":
    main()