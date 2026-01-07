#!/usr/bin/env python3
"""
Multi-Observer Complementarity

The core claim of the aperture framework: same dynamics, different observers,
different experienced time. This simulation demonstrates complementarity
by placing 20 observers at different radii and showing how their clocks diverge.

Key visualization:
- One underlying dynamical system
- 20 observers with apertures closing at different rates
- Their accumulated Fisher time τ diverges dramatically
- The infalling observer (constant aperture) sees linear time
- External observers see time freeze as they approach the horizon

This is the honest demonstration of complementarity without GR.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from dataclasses import dataclass
import os

os.makedirs('../figures', exist_ok=True)


@dataclass
class Config:
    N: int = 100              # Oscillators
    n_observers: int = 20     # Number of observers at different radii
    n_steps: int = 8000       # Integration steps
    dt: float = 0.01
    kappa: float = 0.1
    gamma: float = 0.01
    r_s: float = 0.05         # Horizon radius


def coupled_oscillator_rhs(state, t, N, kappa, gamma):
    x = state[:N]
    p = state[N:]
    dx = p
    dp = -x - gamma * p + kappa * (np.roll(x, 1) - 2*x + np.roll(x, -1))
    return np.concatenate([dx, dp])


def compute_aperture_weights(N: int, r: float, r_s: float) -> np.ndarray:
    """Aperture weights as function of radius."""
    f = np.linspace(0, 1, N)
    effective_r = max(r - r_s, 1e-10)
    weights = np.exp(-f * r_s / effective_r)
    return weights


def compute_tau_dot(velocities: np.ndarray, weights: np.ndarray) -> float:
    """Fisher-speed time proxy."""
    return np.sqrt(np.sum(weights * velocities**2))


def simulate_observers(cfg: Config):
    """
    Run simulation with multiple observers at different radii.
    """
    # Observer radii: log-spaced from near-horizon to far
    observer_radii = np.logspace(np.log10(cfg.r_s * 1.1), 0, cfg.n_observers)

    # Add infalling observer (constant aperture = 1)
    observer_radii = np.append(observer_radii, [1.0])  # Infalling = full access

    print(f"Simulating {len(observer_radii)} observers...")

    # Integrate dynamics once
    np.random.seed(42)
    x0 = np.random.randn(cfg.N)
    p0 = np.random.randn(cfg.N)
    state0 = np.concatenate([x0, p0])

    t = np.linspace(0, cfg.n_steps * cfg.dt, cfg.n_steps)
    solution = odeint(coupled_oscillator_rhs, state0, t,
                      args=(cfg.N, cfg.kappa, cfg.gamma))

    # Compute τ for each observer
    tau_accumulated = {r: np.zeros(cfg.n_steps) for r in observer_radii}
    tau_dot_series = {r: np.zeros(cfg.n_steps) for r in observer_radii}

    for r in observer_radii:
        weights = compute_aperture_weights(cfg.N, r, cfg.r_s)

        for i in range(cfg.n_steps):
            velocities = solution[i, cfg.N:]
            tau_dot = compute_tau_dot(velocities, weights)
            tau_dot_series[r][i] = tau_dot
            if i > 0:
                tau_accumulated[r][i] = tau_accumulated[r][i-1] + tau_dot * cfg.dt
            else:
                tau_accumulated[r][i] = 0

    return observer_radii, tau_accumulated, tau_dot_series, t


def plot_complementarity(observer_radii, tau_accumulated, tau_dot_series, t, cfg):
    """
    Main visualization of complementarity.
    """
    fig = plt.figure(figsize=(14, 10))

    # Color map based on radius
    norm = Normalize(vmin=cfg.r_s, vmax=1.0)
    cmap = cm.viridis

    # --- Panel A: Accumulated τ over coordinate time ---
    ax1 = fig.add_subplot(2, 2, 1)

    for r in sorted(observer_radii[:-1]):  # All except infalling
        color = cmap(norm(r))
        ax1.plot(t, tau_accumulated[r], color=color, lw=1.5, alpha=0.7)

    # Infalling observer (dashed, black)
    ax1.plot(t, tau_accumulated[observer_radii[-1]], 'k--', lw=2,
             label='Infalling observer')

    ax1.set_xlabel('Coordinate time t')
    ax1.set_ylabel('Accumulated Fisher time τ')
    ax1.set_title('(A) Same dynamics, different clocks')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, label='Observer radius r')

    # --- Panel B: τ̇ vs radius at final time ---
    ax2 = fig.add_subplot(2, 2, 2)

    final_tau_dot = [np.mean(tau_dot_series[r][-1000:]) for r in observer_radii[:-1]]
    external_radii = observer_radii[:-1]

    ax2.semilogy(external_radii, final_tau_dot, 'C0o-', lw=2, markersize=6)
    ax2.axvline(cfg.r_s, color='k', ls=':', label=f'Horizon r_s = {cfg.r_s}')

    # Schwarzschild prediction
    r_fine = np.linspace(cfg.r_s * 1.01, 1.0, 100)
    schw = np.sqrt(1 - cfg.r_s / r_fine)
    schw_scaled = schw * max(final_tau_dot) / max(schw)
    ax2.semilogy(r_fine, schw_scaled, 'C1--', lw=2, label='Schwarzschild shape')

    ax2.set_xlabel('Observer radius r')
    ax2.set_ylabel('Time rate τ̇ (log scale)')
    ax2.set_title('(B) Clock rate vs radius')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Panel C: Time dilation ratio ---
    ax3 = fig.add_subplot(2, 2, 3)

    infalling_tau = tau_accumulated[observer_radii[-1]][-1]
    dilation_ratios = [infalling_tau / max(tau_accumulated[r][-1], 1e-10)
                       for r in observer_radii[:-1]]

    ax3.semilogy(external_radii, dilation_ratios, 'C3o-', lw=2, markersize=6)
    ax3.axhline(1.0, color='k', ls=':', alpha=0.5)
    ax3.axvline(cfg.r_s, color='k', ls=':', label=f'Horizon')

    ax3.set_xlabel('Observer radius r')
    ax3.set_ylabel('Time dilation ratio τ_inf / τ_obs')
    ax3.set_title('(C) Time dilation relative to infalling observer')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Annotate max dilation
    max_dilation = max(dilation_ratios)
    ax3.text(0.95, 0.95, f'Max dilation: {max_dilation:.1f}×',
             transform=ax3.transAxes, ha='right', va='top', fontsize=12)

    # --- Panel D: Snapshot of τ̇ across observers ---
    ax4 = fig.add_subplot(2, 2, 4)

    # Take 5 time slices
    time_slices = [1000, 2000, 4000, 6000, 7500]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(time_slices)))

    for ti, c in zip(time_slices, colors):
        tau_dots_at_t = [tau_dot_series[r][ti] for r in observer_radii[:-1]]
        ax4.semilogy(external_radii, tau_dots_at_t, 'o-', color=c, lw=1.5,
                     markersize=4, label=f't = {t[ti]:.1f}', alpha=0.8)

    ax4.axvline(cfg.r_s, color='k', ls=':', alpha=0.5)
    ax4.set_xlabel('Observer radius r')
    ax4.set_ylabel('Instantaneous τ̇')
    ax4.set_title('(D) Clock rates at different times')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/fig_complementarity.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('../figures/fig_complementarity.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Summary statistics
    print(f"\nComplementarity summary:")
    print(f"  Number of observers: {len(observer_radii)}")
    print(f"  Infalling τ: {tau_accumulated[observer_radii[-1]][-1]:.1f}")
    print(f"  Near-horizon τ: {tau_accumulated[observer_radii[0]][-1]:.1f}")
    print(f"  Max time dilation: {max_dilation:.1f}×")
    print(f"  Saved: fig_complementarity.pdf")


if __name__ == "__main__":
    cfg = Config()

    print("=" * 60)
    print("MULTI-OBSERVER COMPLEMENTARITY SIMULATION")
    print("=" * 60)

    observer_radii, tau_accumulated, tau_dot_series, t = simulate_observers(cfg)
    plot_complementarity(observer_radii, tau_accumulated, tau_dot_series, t, cfg)

    print("\nDone!")
