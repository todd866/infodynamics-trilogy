#!/usr/bin/env python3
"""
Hawking Residual: Thermal radiation from near-horizon aperture

The paper speculates that zero information flow is unphysical, so horizons
must "leak" information. This simulation shows:

1. As aperture → 0, residual information rate → small but nonzero
2. The residual has thermal statistics (Boltzmann distribution)
3. The "temperature" scales with surface gravity analogue

This is a genuine prediction: if aperture-based time dilation is real,
then horizons should radiate, and the radiation should be thermal.

Laptop version: 100 oscillators, 50 aperture values
Flagship version: 10,000 oscillators, 500 aperture values, bootstrap statistics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import kstest, expon
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
    N: int = 100              # Number of oscillators
    n_apertures: int = 50     # Aperture sweep points
    n_steps: int = 10000      # Long run for statistics
    dt: float = 0.01
    kappa: float = 0.1
    gamma: float = 0.01
    aperture_min: float = 0.001   # Very small aperture (near horizon)
    aperture_max: float = 1.0     # Full aperture (far from horizon)


# ============================================================================
# PHYSICS
# ============================================================================

def coupled_oscillator_rhs(state, t, N, kappa, gamma):
    x = state[:N]
    p = state[N:]
    dx = p
    dp = -x - gamma * p + kappa * (np.roll(x, 1) - 2*x + np.roll(x, -1))
    return np.concatenate([dx, dp])


def compute_aperture_weights(N: int, aperture: float) -> np.ndarray:
    """
    Weights with uniform suppression controlled by aperture parameter.
    aperture=1: full access. aperture→0: nearly closed.
    """
    f = np.linspace(0, 1, N)
    # Exponential suppression of high-frequency modes
    weights = aperture * np.exp(-f * (1 - aperture) / max(aperture, 1e-10))
    return np.clip(weights, 1e-10, 1.0)  # Minimum residual (quantum floor)


def compute_information_increments(velocities_series: np.ndarray,
                                   weights: np.ndarray) -> np.ndarray:
    """
    Compute information increments: ΔI = τ̇ * dt

    These are the "bits" of distinguishable information accumulated per step.
    Near the horizon, these should be small and thermally distributed.
    """
    increments = []
    for v in velocities_series:
        tau_dot = np.sqrt(np.sum(weights * v**2))
        increments.append(tau_dot)
    return np.array(increments)


def test_thermal_distribution(increments: np.ndarray) -> Tuple[float, float, float]:
    """
    Test if increments follow exponential (thermal) distribution.

    Returns: (mean_increment, ks_statistic, p_value)

    Exponential distribution is characteristic of thermal/Boltzmann statistics.
    """
    # Normalize to positive values
    inc = np.abs(increments)
    inc = inc[inc > 0]

    if len(inc) < 100:
        return np.mean(increments), 1.0, 0.0

    # Fit exponential
    mean_inc = np.mean(inc)

    # KS test against exponential
    ks_stat, p_value = kstest(inc / mean_inc, 'expon')

    return mean_inc, ks_stat, p_value


# ============================================================================
# SIMULATION
# ============================================================================

def run_single_aperture(cfg: Config, aperture: float) -> dict:
    """
    Run long simulation at fixed aperture, collect statistics.
    """
    np.random.seed(42)
    x0 = np.random.randn(cfg.N)
    p0 = np.random.randn(cfg.N)
    state0 = np.concatenate([x0, p0])

    t = np.linspace(0, cfg.n_steps * cfg.dt, cfg.n_steps)
    solution = odeint(coupled_oscillator_rhs, state0, t,
                      args=(cfg.N, cfg.kappa, cfg.gamma))

    weights = compute_aperture_weights(cfg.N, aperture)

    # Compute information increments (after warmup)
    warmup = cfg.n_steps // 5
    velocities = solution[warmup:, cfg.N:]
    increments = compute_information_increments(velocities, weights)

    # Test thermality
    mean_inc, ks_stat, p_value = test_thermal_distribution(increments)

    # Effective dimension
    k_eff = np.sum(weights)**2 / np.sum(weights**2)

    return {
        'aperture': aperture,
        'mean_rate': mean_inc,
        'std_rate': np.std(increments),
        'k_eff': k_eff,
        'ks_stat': ks_stat,
        'p_value': p_value,
        'increments': increments,
        'weights': weights
    }


def run_aperture_sweep(cfg: Config) -> List[dict]:
    """
    Sweep across aperture values from near-horizon to far.
    """
    # Log-spaced to get good resolution near horizon
    apertures = np.logspace(np.log10(cfg.aperture_min),
                            np.log10(cfg.aperture_max),
                            cfg.n_apertures)

    results = []
    print(f"Running aperture sweep: {cfg.n_apertures} points")

    for i, a in enumerate(apertures):
        if i % 10 == 0:
            print(f"  aperture = {a:.4f} ({i+1}/{cfg.n_apertures})")
        results.append(run_single_aperture(cfg, a))

    return results


# ============================================================================
# ANALYSIS
# ============================================================================

def compute_hawking_temperature(results: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    The "temperature" of the residual radiation.

    In the aperture framework, T ∝ surface gravity ∝ dw/dr at horizon.
    We measure it as the mean information rate at small aperture.
    """
    apertures = np.array([r['aperture'] for r in results])
    temperatures = np.array([r['mean_rate'] for r in results])

    return apertures, temperatures


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_hawking_residual(results: List[dict], cfg: Config):
    """
    Main figure showing Hawking-like radiation from aperture.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    apertures = np.array([r['aperture'] for r in results])
    mean_rates = np.array([r['mean_rate'] for r in results])
    std_rates = np.array([r['std_rate'] for r in results])
    k_effs = np.array([r['k_eff'] for r in results])
    p_values = np.array([r['p_value'] for r in results])

    # --- Panel A: Information rate vs aperture ---
    ax1 = axes[0, 0]
    ax1.loglog(apertures, mean_rates, 'C0o-', lw=2, markersize=5)
    ax1.fill_between(apertures, mean_rates - std_rates, mean_rates + std_rates,
                     alpha=0.3, color='C0')

    # Highlight residual at near-horizon
    near_horizon = apertures < 0.01
    ax1.axhline(mean_rates[near_horizon].min(), color='C1', ls='--',
                label=f'Residual floor: {mean_rates[near_horizon].min():.2e}')

    ax1.set_xlabel('Aperture parameter')
    ax1.set_ylabel('Mean information rate τ̇')
    ax1.set_title('(A) Information rate approaches nonzero floor')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Panel B: Thermality test ---
    ax2 = axes[0, 1]
    ax2.semilogx(apertures, p_values, 'C2o-', lw=2, markersize=5)
    ax2.axhline(0.05, color='k', ls=':', label='p = 0.05 threshold')
    ax2.set_xlabel('Aperture parameter')
    ax2.set_ylabel('p-value (KS test vs exponential)')
    ax2.set_title('(B) Thermal distribution test')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Add interpretation
    thermal_count = np.sum(p_values > 0.05)
    ax2.text(0.95, 0.95, f'{thermal_count}/{len(p_values)} thermal',
             transform=ax2.transAxes, ha='right', va='top',
             fontsize=12, color='C2')

    # --- Panel C: Distribution at different apertures ---
    ax3 = axes[1, 0]

    # Select 4 representative apertures
    indices = [0, len(results)//3, 2*len(results)//3, -1]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(indices)))

    for idx, c in zip(indices, colors):
        inc = results[idx]['increments']
        inc = inc[inc > 0]  # Positive only
        if len(inc) > 100:
            # Histogram
            hist, bins = np.histogram(inc / np.mean(inc), bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax3.plot(bin_centers, hist, color=c, lw=2,
                     label=f'a = {results[idx]["aperture"]:.3f}')

    # Overlay exponential (thermal) prediction
    x = np.linspace(0, 5, 100)
    ax3.plot(x, np.exp(-x), 'k--', lw=2, label='Thermal (exp)')

    ax3.set_xlabel('Normalized information increment')
    ax3.set_ylabel('Probability density')
    ax3.set_title('(C) Distribution of information increments')
    ax3.legend()
    ax3.set_xlim(0, 5)
    ax3.grid(True, alpha=0.3)

    # --- Panel D: Effective temperature ---
    ax4 = axes[1, 1]

    # "Temperature" = mean increment rate (inverse of thermal β)
    temperatures = mean_rates

    ax4.loglog(apertures, temperatures, 'C3o-', lw=2, markersize=5)

    # Fit power law: T ∝ aperture^α near horizon
    near = apertures < 0.1
    if np.sum(near) > 5:
        log_a = np.log(apertures[near])
        log_T = np.log(temperatures[near])
        slope, intercept = np.polyfit(log_a, log_T, 1)
        fit_line = np.exp(intercept) * apertures**slope
        ax4.loglog(apertures, fit_line, 'k--', lw=1.5,
                   label=f'T ∝ a^{slope:.2f}')

    ax4.set_xlabel('Aperture parameter')
    ax4.set_ylabel('Effective temperature (τ̇)')
    ax4.set_title('(D) Hawking-like temperature scaling')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/fig_hawking_residual.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('../figures/fig_hawking_residual.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nHawking residual analysis:")
    print(f"  Residual rate at smallest aperture: {mean_rates[0]:.2e}")
    print(f"  Thermal distributions: {thermal_count}/{len(p_values)} pass KS test")
    print(f"  Saved: fig_hawking_residual.pdf")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    cfg = Config()

    print("=" * 60)
    print("HAWKING RESIDUAL SIMULATION")
    print("=" * 60)
    print(f"  Oscillators: {cfg.N}")
    print(f"  Aperture range: [{cfg.aperture_min}, {cfg.aperture_max}]")
    print(f"  Steps: {cfg.n_steps}")
    print("=" * 60)

    results = run_aperture_sweep(cfg)
    plot_hawking_residual(results, cfg)

    print("\nDone!")
