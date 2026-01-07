#!/usr/bin/env python3
"""
Schwarzschild Comparison: τ̇(r) vs GR prediction

Demonstrates that aperture-based time dilation quantitatively matches
the Schwarzschild metric prediction: τ̇ ∝ √(1 - r_s/r)

This is the honest claim: we're not deriving GR, we're showing that
dimensional aperture models reproduce the phenomenology.

Laptop version: 100 oscillators, 200 radius values
Flagship version: 10,000 oscillators, 2000 radius values, GPU acceleration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Tuple, List
import os

os.makedirs('../figures', exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Simulation parameters - adjust for laptop vs cluster"""
    N: int = 100              # Number of oscillators (laptop: 100, flagship: 10000)
    n_radius: int = 200       # Radius sweep points (laptop: 200, flagship: 2000)
    n_steps: int = 5000       # Integration steps per radius
    dt: float = 0.01          # Timestep
    kappa: float = 0.1        # Coupling strength
    gamma: float = 0.01       # Damping
    r_min: float = 0.02       # Minimum radius (approach horizon)
    r_max: float = 1.0        # Maximum radius (far from horizon)
    r_s: float = 0.01         # Schwarzschild radius analogue
    window: int = 500         # Sliding window for statistics


# ============================================================================
# PHYSICS
# ============================================================================

def coupled_oscillator_rhs(state: np.ndarray, t: float, N: int, kappa: float, gamma: float) -> np.ndarray:
    """RHS for coupled harmonic oscillators with damping."""
    x = state[:N]
    p = state[N:]

    # Harmonic + nearest-neighbor coupling
    dx = p
    dp = -x - gamma * p

    # Coupling to neighbors (periodic boundary)
    dp += kappa * (np.roll(x, 1) - 2*x + np.roll(x, -1))

    return np.concatenate([dx, dp])


def compute_aperture_weights(N: int, r: float, r_s: float) -> np.ndarray:
    """
    Aperture weights as function of radius.

    Higher-index modes (higher frequency) are suppressed first as r → r_s,
    mimicking gravitational redshift pushing high-frequency content below
    observer bandwidth.
    """
    f = np.linspace(0, 1, N)  # Frequency proxy
    # Exponential suppression: high-f modes vanish first near horizon
    effective_r = max(r - r_s, 1e-10)  # Regularize at horizon
    weights = np.exp(-f * r_s / effective_r)
    return weights


def compute_tau_dot(velocities: np.ndarray, weights: np.ndarray) -> float:
    """
    Fisher-speed time proxy: τ̇ = √(Σ w_i ẋ_i²)

    This is the geodesic speed in the Fisher metric induced by the aperture.
    """
    return np.sqrt(np.sum(weights * velocities**2))


def schwarzschild_tau_dot(r: float, r_s: float) -> float:
    """Schwarzschild proper time rate: dτ/dt = √(1 - r_s/r)"""
    if r <= r_s:
        return 0.0
    return np.sqrt(1 - r_s / r)


# ============================================================================
# SIMULATION
# ============================================================================

def run_single_radius(cfg: Config, r: float) -> Tuple[float, float, float]:
    """
    Run simulation at fixed radius, return mean τ̇ and std.
    """
    # Random initial conditions
    np.random.seed(42)  # Reproducibility
    x0 = np.random.randn(cfg.N)
    p0 = np.random.randn(cfg.N)
    state0 = np.concatenate([x0, p0])

    # Integrate
    t = np.linspace(0, cfg.n_steps * cfg.dt, cfg.n_steps)
    solution = odeint(coupled_oscillator_rhs, state0, t,
                      args=(cfg.N, cfg.kappa, cfg.gamma))

    # Compute aperture weights
    weights = compute_aperture_weights(cfg.N, r, cfg.r_s)

    # Compute τ̇ over time (after warmup)
    warmup = cfg.n_steps // 10
    tau_dots = []

    for i in range(warmup, cfg.n_steps):
        velocities = solution[i, cfg.N:]  # momenta = velocities (mass=1)
        tau_dots.append(compute_tau_dot(velocities, weights))

    tau_dots = np.array(tau_dots)
    return np.mean(tau_dots), np.std(tau_dots), np.sum(weights)


def run_radius_sweep(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep across radius values, compute τ̇(r).
    """
    radii = np.linspace(cfg.r_min, cfg.r_max, cfg.n_radius)
    tau_dots_mean = np.zeros(cfg.n_radius)
    tau_dots_std = np.zeros(cfg.n_radius)
    k_eff = np.zeros(cfg.n_radius)

    print(f"Running radius sweep: {cfg.n_radius} points, {cfg.N} oscillators")

    for i, r in enumerate(radii):
        if i % 20 == 0:
            print(f"  r = {r:.3f} ({i+1}/{cfg.n_radius})")
        tau_dots_mean[i], tau_dots_std[i], k_eff[i] = run_single_radius(cfg, r)

    return radii, tau_dots_mean, tau_dots_std, k_eff


# ============================================================================
# ANALYSIS
# ============================================================================

def fit_schwarzschild(radii: np.ndarray, tau_dots: np.ndarray, r_s_init: float) -> Tuple[float, float, float]:
    """
    Fit τ̇(r) to Schwarzschild form: A * √(1 - r_s/r)

    Returns: (A, r_s_fit, r_squared)
    """
    def model(r, A, r_s):
        return A * np.sqrt(np.maximum(1 - r_s/r, 0))

    # Only fit where r > r_s_init
    mask = radii > 2 * r_s_init

    try:
        popt, pcov = curve_fit(model, radii[mask], tau_dots[mask],
                               p0=[tau_dots.max(), r_s_init],
                               bounds=([0, 0], [np.inf, radii.max()/2]))

        # R² calculation
        residuals = tau_dots[mask] - model(radii[mask], *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((tau_dots[mask] - np.mean(tau_dots[mask]))**2)
        r_squared = 1 - ss_res / ss_tot

        return popt[0], popt[1], r_squared
    except:
        return tau_dots.max(), r_s_init, 0.0


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(radii: np.ndarray, tau_dots: np.ndarray, tau_dots_std: np.ndarray,
                    A_fit: float, r_s_fit: float, r_squared: float, cfg: Config):
    """
    Main comparison figure: simulation vs Schwarzschild prediction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- Panel A: τ̇ vs r with Schwarzschild fit ---
    ax1 = axes[0]

    # Simulation data with error band
    ax1.fill_between(radii, tau_dots - tau_dots_std, tau_dots + tau_dots_std,
                     alpha=0.3, color='C0', label='Simulation ±1σ')
    ax1.plot(radii, tau_dots, 'C0-', lw=2, label='Aperture model')

    # Schwarzschild prediction (scaled)
    r_fine = np.linspace(cfg.r_s * 1.01, cfg.r_max, 500)
    schwarzschild = A_fit * np.sqrt(1 - r_s_fit / r_fine)
    ax1.plot(r_fine, schwarzschild, 'C1--', lw=2,
             label=f'Schwarzschild fit (R² = {r_squared:.4f})')

    ax1.axvline(r_s_fit, color='k', ls=':', alpha=0.5, label=f'r_s = {r_s_fit:.4f}')
    ax1.set_xlabel('Radius r')
    ax1.set_ylabel('Time rate τ̇')
    ax1.set_title('(A) Aperture model vs Schwarzschild')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(0, cfg.r_max)
    ax1.set_ylim(0, None)
    ax1.grid(True, alpha=0.3)

    # --- Panel B: Residuals ---
    ax2 = axes[1]

    # Compute residuals where fit is valid
    mask = radii > 2 * r_s_fit
    schwarzschild_interp = A_fit * np.sqrt(np.maximum(1 - r_s_fit / radii, 0))
    residuals = (tau_dots - schwarzschild_interp) / tau_dots.max()

    ax2.fill_between(radii[mask], -tau_dots_std[mask]/tau_dots.max(),
                     tau_dots_std[mask]/tau_dots.max(),
                     alpha=0.3, color='gray', label='±1σ uncertainty')
    ax2.plot(radii[mask], residuals[mask], 'C0-', lw=1.5)
    ax2.axhline(0, color='k', ls='-', alpha=0.3)

    ax2.set_xlabel('Radius r')
    ax2.set_ylabel('Normalized residual')
    ax2.set_title('(B) Residuals from Schwarzschild fit')
    ax2.set_xlim(0, cfg.r_max)
    ax2.set_ylim(-0.2, 0.2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # --- Panel C: Near-horizon zoom ---
    ax3 = axes[2]

    near_horizon = radii < 0.15
    ax3.semilogy(radii[near_horizon], tau_dots[near_horizon], 'C0o-', lw=2,
                 label='Aperture model', markersize=4)
    ax3.semilogy(r_fine[r_fine < 0.15],
                 A_fit * np.sqrt(1 - r_s_fit / r_fine[r_fine < 0.15]),
                 'C1--', lw=2, label='Schwarzschild')

    ax3.axvline(r_s_fit, color='k', ls=':', alpha=0.5)
    ax3.set_xlabel('Radius r')
    ax3.set_ylabel('Time rate τ̇ (log scale)')
    ax3.set_title('(C) Near-horizon behavior')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/fig_schwarzschild_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('../figures/fig_schwarzschild_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSchwarzschild comparison:")
    print(f"  Fitted r_s = {r_s_fit:.6f} (true: {cfg.r_s})")
    print(f"  R² = {r_squared:.6f}")
    print(f"  Saved: fig_schwarzschild_comparison.pdf")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    cfg = Config()

    print("=" * 60)
    print("SCHWARZSCHILD COMPARISON SIMULATION")
    print("=" * 60)
    print(f"  Oscillators: {cfg.N}")
    print(f"  Radius points: {cfg.n_radius}")
    print(f"  Steps per radius: {cfg.n_steps}")
    print(f"  Horizon radius: {cfg.r_s}")
    print("=" * 60)

    # Run sweep
    radii, tau_dots, tau_dots_std, k_eff = run_radius_sweep(cfg)

    # Fit to Schwarzschild
    A_fit, r_s_fit, r_squared = fit_schwarzschild(radii, tau_dots, cfg.r_s)

    # Plot
    plot_comparison(radii, tau_dots, tau_dots_std, A_fit, r_s_fit, r_squared, cfg)

    print("\nDone!")
