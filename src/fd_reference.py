"""
Classical 2D Finite-Difference (FD) Reference Solver for the 2D Wave Equation:
    u_tt = c^2 * (u_xx + u_yy)  on  (x, y) in [0, 1]^2, t in [0, T]
with homogeneous Dirichlet boundary conditions u = 0 on dOmega.
"""

import time
from typing import Tuple, Dict
import numpy as np


def initial_displacement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Initial wave profile u(x, y, 0): asymmetric two-mode standing wave pulse.
    Satisfies Dirichlet boundaries u = 0 at x in {0, 1} and y in {0, 1}.
    """
    return (
        np.sin(np.pi * x) * np.sin(np.pi * y)
        + 0.5 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)
    )


def initial_velocity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Initial velocity field u_t(x, y, 0) = 0."""
    return np.zeros_like(x)


def analytical_solution_grid(
    x_grid: np.ndarray, y_grid: np.ndarray, t_grid: np.ndarray, c: float = 1.0
) -> np.ndarray:
    """
    Exact closed-form solution evaluated across the full (nt, nx, ny) space-time grid.
    """
    T, X, Y = np.meshgrid(t_grid, x_grid, y_grid, indexing="ij")
    omega_11 = np.pi * c * np.sqrt(1.0**2 + 1.0**2)
    omega_21 = np.pi * c * np.sqrt(2.0**2 + 1.0**2)
    return (
        np.sin(np.pi * X) * np.sin(np.pi * Y) * np.cos(omega_11 * T)
        + 0.5 * np.sin(2.0 * np.pi * X) * np.sin(np.pi * Y) * np.cos(omega_21 * T)
    )


def _laplacian_2d(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Second-order central difference 5-point 2D Laplacian on interior nodes."""
    lap = np.zeros_like(u)
    u_xx = (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx**2)
    u_yy = (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy**2)
    lap[1:-1, 1:-1] = u_xx + u_yy
    return lap


def _run_fd_time_marching(
    X: np.ndarray,
    Y: np.ndarray,
    nt: int,
    dx: float,
    dy: float,
    dt: float,
    c: float,
    use_taylor_first_step: bool = True,
) -> np.ndarray:
    """Core explicit leapfrog time-marching loop with selectable n=1 initialization."""
    nx, ny = X.shape
    u_fd = np.zeros((nt, nx, ny), dtype=np.float64)

    # Step n = 0: Initial displacement
    u_fd[0] = initial_displacement(X, Y)
    v0 = initial_velocity(X, Y)

    # Step n = 1: Second-order Taylor ghost-point step vs. Naive first-order step
    if use_taylor_first_step:
        lap_0 = _laplacian_2d(u_fd[0], dx, dy)
        u_fd[1] = u_fd[0] + dt * v0 + 0.5 * (c * dt) ** 2 * lap_0
    else:
        u_fd[1] = u_fd[0] + dt * v0

    u_fd[1, 0, :] = u_fd[1, -1, :] = u_fd[1, :, 0] = u_fd[1, :, -1] = 0.0

    # Steps n = 1 ... nt - 2: Explicit leapfrog time-stepping
    c_dt_sq = (c * dt) ** 2
    for n in range(1, nt - 1):
        lap_n = _laplacian_2d(u_fd[n], dx, dy)
        u_fd[n + 1] = 2.0 * u_fd[n] - u_fd[n - 1] + c_dt_sq * lap_n
        u_fd[n + 1, 0, :] = 0.0
        u_fd[n + 1, -1, :] = 0.0
        u_fd[n + 1, :, 0] = 0.0
        u_fd[n + 1, :, -1] = 0.0

    return u_fd


def solve_fd_wave_2d(
    nx: int = 81,
    ny: int = 81,
    nt: int = 251,
    t_max: float = 0.5,
    c: float = 1.0,
    use_taylor_first_step: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Solves the 2D wave equation using a second-order explicit central difference scheme
    and audits both the Taylor-step and naive-step discretization errors against the
    exact analytical solution.
    """
    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    t_grid = np.linspace(0.0, t_max, nt)

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dt = t_grid[1] - t_grid[0]

    cfl = c * dt * np.sqrt((1.0 / dx**2) + (1.0 / dy**2))
    if cfl >= 1.0:
        raise ValueError(f"FD scheme is unstable: CFL = {cfl:.4f} >= 1.0.")

    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    # Time the primary FD solve
    t0 = time.perf_counter()
    u_fd = _run_fd_time_marching(
        X, Y, nt, dx, dy, dt, c, use_taylor_first_step=use_taylor_first_step
    )
    fd_wall_clock_ms = (time.perf_counter() - t0) * 1000.0

    # Evaluate exact analytical solution across the 3D grid
    u_exact = analytical_solution_grid(x_grid, y_grid, t_grid, c=c)
    exact_norm_full = np.linalg.norm(u_exact)
    exact_norm_final = np.linalg.norm(u_exact[-1])

    fd_rel_l2_full = np.linalg.norm(u_fd - u_exact) / exact_norm_full
    fd_rel_l2_final = np.linalg.norm(u_fd[-1] - u_exact[-1]) / exact_norm_final

    # Compute naive first-step ablation (Section 4.3 reproducibility check)
    u_fd_naive = _run_fd_time_marching(
        X, Y, nt, dx, dy, dt, c, use_taylor_first_step=False
    )
    naive_rel_l2_full = np.linalg.norm(u_fd_naive - u_exact) / exact_norm_full
    naive_rel_l2_final = np.linalg.norm(u_fd_naive[-1] - u_exact[-1]) / exact_norm_final

    meta = {
        "dx": float(dx),
        "dy": float(dy),
        "dt": float(dt),
        "cfl": float(cfl),
        "fd_wall_clock_ms": round(float(fd_wall_clock_ms), 2),
        "fd_vs_exact_rel_l2_full": float(fd_rel_l2_full),
        "fd_vs_exact_rel_l2_final": float(fd_rel_l2_final),
        "fd_naive_vs_exact_rel_l2_full": float(naive_rel_l2_full),
        "fd_naive_vs_exact_rel_l2_final": float(naive_rel_l2_final),
        "fd_taylor_improvement_factor": round(
            float(naive_rel_l2_full / fd_rel_l2_full), 2
        ),
    }
    return x_grid, y_grid, t_grid, u_fd, u_exact, meta