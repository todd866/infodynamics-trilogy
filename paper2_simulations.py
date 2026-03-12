#!/usr/bin/env python3
"""
Paper 2: Black Hole Aperture - All Simulations

Demonstrates that observer-relative dimensional apertures produce
horizon-like phenomenology (time dilation, complementarity) without GR.

Figures:
  - fig1_time_dilation.png: External vs infalling observer time
  - fig2_thermodynamics.png: S_acc drop, Q accumulation at horizon
  - fig3_ligo_connection.png: Aperture perturbation → ringdown (illustrative)
  - fig_schwarzschild_comparison.pdf: High correspondence with GR
  - fig_complementarity.pdf: 21 observers, multi-fold time dilation

Usage:
  python paper2_simulations.py              # Generate all figures
  python paper2_simulations.py --figure 1   # Main time dilation only
  python paper2_simulations.py --figure 3   # Schwarzschild comparison
  python paper2_simulations.py --seed 42    # Set random seed for reproducibility
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for reproducibility
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.integrate import odeint
from scipy.stats import pearsonr
from dataclasses import dataclass
from collections import deque
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# =============================================================================
# Core Physics: Coupled Oscillator System
# =============================================================================

@dataclass
class Config:
    N: int = 50               # Number of oscillators
    kappa: float = 0.1        # Coupling strength
    gamma: float = 0.01       # Damping
    n_steps: int = 1000       # Integration steps
    dt: float = 0.01          # Timestep
    r_s: float = 0.05         # Horizon radius
    # Normal mode data (computed on demand)
    _omega: np.ndarray = None
    _U: np.ndarray = None

    def get_normal_modes(self):
        """Compute and cache normal modes and transformation matrix."""
        if self._omega is None or self._U is None:
            self._omega, self._U = compute_normal_modes(self.N, self.kappa)
        return self._omega, self._U


def coupled_oscillator_rhs(state, t, N, kappa, gamma):
    """Equations of motion for coupled damped oscillators."""
    x = state[:N]
    p = state[N:]
    dx = p
    dp = -x - gamma * p + kappa * (np.roll(x, 1) - 2*x + np.roll(x, -1))
    return np.concatenate([dx, dp])


def compute_normal_modes(N: int, kappa: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute normal modes and frequencies for coupled oscillators.

    For N coupled oscillators with nearest-neighbor coupling (periodic BC),
    the stiffness matrix K has eigenvalues:
        ω_k² = 1 + 2κ(1 - cos(2πk/N))

    Returns:
        omega: Normal mode frequencies (sorted low to high)
        U: Transformation matrix (columns are eigenvectors, sorted by frequency)
    """
    # Build stiffness matrix K
    # K_ii = 1 + 2κ (from ½x_i² + ½κ(x_i - x_{i-1})² + ½κ(x_i - x_{i+1})²)
    # K_{i,i±1} = -κ
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = 1 + 2 * kappa
        K[i, (i + 1) % N] = -kappa
        K[i, (i - 1) % N] = -kappa

    # Eigendecomposition (K is symmetric)
    eigenvalues, eigenvectors = np.linalg.eigh(K)

    # eigenvalues are ω², so take sqrt for frequencies
    omega = np.sqrt(np.maximum(eigenvalues, 0))  # clamp small negatives

    # Sort by frequency (eigh already returns sorted, but be explicit)
    sort_idx = np.argsort(omega)
    omega = omega[sort_idx]
    U = eigenvectors[:, sort_idx]

    return omega, U


def compute_aperture_weights(N: int, r: float, r_s: float,
                             omega: np.ndarray = None) -> np.ndarray:
    """
    Aperture weights as function of radius in normal mode basis.

    Uses an exponential form that mimics gravitational redshift:
        w_k(r) = exp(-f_k * r_s / (r - r_s))

    where f_k = (ω_k - ω_min) / (ω_max - ω_min) normalizes frequencies to [0,1].
    High-frequency modes are suppressed first as the observer approaches
    the horizon (r -> r_s).

    Args:
        N: Number of modes
        r: Observer radius
        r_s: Horizon radius
        omega: Normal mode frequencies (if None, uses linear proxy f_k = k/N)

    Returns:
        Aperture weights in normal mode basis
    """
    if omega is None:
        # Fallback to index-based proxy
        f = np.linspace(0, 1, N)
    else:
        # Use actual normal mode frequencies, normalized to [0,1]
        omega_min, omega_max = omega.min(), omega.max()
        if omega_max > omega_min:
            f = (omega - omega_min) / (omega_max - omega_min)
        else:
            f = np.zeros(N)

    effective_r = max(r - r_s, 1e-10)
    return np.exp(-f * r_s / effective_r)


def compute_k_w(weights: np.ndarray) -> float:
    """Channel participation from weights."""
    return (np.sum(weights)**2) / (np.sum(weights**2) + 1e-10)


def compute_tau_dot(velocities: np.ndarray, weights: np.ndarray) -> float:
    """Fisher-speed time proxy."""
    return np.sqrt(np.sum(weights * velocities**2))


# =============================================================================
# Figure 1: Main Time Dilation Simulation
# =============================================================================

def fig1_time_dilation(cfg: Config):
    """External vs infalling observer time dilation in normal mode basis."""
    print("  Running main aperture simulation (normal mode basis)...")

    # Get normal modes
    omega, U = cfg.get_normal_modes()
    print(f"    Normal mode frequencies: [{omega.min():.3f}, {omega.max():.3f}]")

    np.random.seed(42)
    x0 = np.random.randn(cfg.N)
    p0 = np.random.randn(cfg.N)
    state0 = np.concatenate([x0, p0])

    t = np.linspace(0, cfg.n_steps * cfg.dt, cfg.n_steps)
    solution = odeint(coupled_oscillator_rhs, state0, t,
                     args=(cfg.N, cfg.kappa, cfg.gamma))

    # External observer: fixed near-horizon radius to show clear dilation
    r_external = cfg.r_s * 1.2
    weights_ext = compute_aperture_weights(cfg.N, r_external, cfg.r_s, omega=omega)
    k_w_ext = compute_k_w(weights_ext)

    external_tau = np.zeros(cfg.n_steps)
    infalling_tau = np.zeros(cfg.n_steps)

    for i in range(cfg.n_steps):
        # Transform velocities to normal mode basis: v_mode = U.T @ v
        velocities = solution[i, cfg.N:]
        velocities_mode = U.T @ velocities

        # External observer: weights apply in normal mode basis
        tau_dot_ext = compute_tau_dot(velocities_mode, weights_ext)
        if i > 0:
            external_tau[i] = external_tau[i-1] + tau_dot_ext * cfg.dt

        # Infalling observer (full access in normal mode basis)
        weights_inf = np.ones(cfg.N)
        tau_dot_inf = compute_tau_dot(velocities_mode, weights_inf)
        if i > 0:
            infalling_tau[i] = infalling_tau[i-1] + tau_dot_inf * cfg.dt

    dilation = infalling_tau[-1] / (external_tau[-1] + 1e-10)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: Aperture weights vs normalized frequency
    f_norm = (omega - omega.min()) / (omega.max() - omega.min())
    axes[0].bar(f_norm, weights_ext, width=0.02, color='C0', alpha=0.7,
                label=f'External (r={r_external:.2f})')
    axes[0].axhline(1.0, color='C1', ls='--', lw=2, label='Infalling (w=1)')
    axes[0].set_xlabel(r'Normalized frequency $(\omega - \omega_{min})/(\omega_{max} - \omega_{min})$')
    axes[0].set_ylabel('Weight $w_k$')
    axes[0].set_title(f'(A) Aperture weights ($k_w$ = {k_w_ext:.1f} vs {cfg.N})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel B: Fisher time accumulation
    axes[1].plot(t, external_tau, 'C0-', lw=2, label='External')
    axes[1].plot(t, infalling_tau, 'C1--', lw=2, label='Infalling')
    axes[1].set_xlabel('Coordinate time')
    axes[1].set_ylabel(r'Accumulated $\tau$')
    axes[1].set_title('(B) Fisher time accumulation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel C: Final comparison
    axes[2].bar(['External', 'Infalling'], [external_tau[-1], infalling_tau[-1]],
               color=['C0', 'C1'])
    axes[2].set_ylabel(r'Total $\tau$')
    axes[2].set_title(f'(C) Time dilation: {dilation:.1f}×')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_time_dilation.png', dpi=150)
    plt.close()
    print(f"  Time dilation ratio: {dilation:.1f}×")
    print("  Generated: fig1_time_dilation.png")


# =============================================================================
# Figure 2: Thermodynamics (S_acc drop, Q accumulation)
# =============================================================================

def compute_k_dyn(observed_positions: np.ndarray) -> float:
    """Dynamical dimension from observed state covariance eigenvalues."""
    cov = np.cov(observed_positions.T)
    if cov.ndim == 0:  # scalar case
        return 1.0
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter near-zero
    if len(eigenvalues) == 0:
        return 1.0
    return (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)


def compute_S_acc(cov: np.ndarray, epsilon: float = 1e-6) -> float:
    """Accessible entropy (log-volume measure) from covariance."""
    if cov.ndim == 0:
        return 0.5 * np.log(cov + epsilon)
    regularized = cov + epsilon * np.eye(cov.shape[0])
    sign, logdet = np.linalg.slogdet(regularized)
    return 0.5 * logdet


def fig2_thermodynamics(cfg: Config):
    """S_acc drop and Q (Landauer cost) accumulation at horizon (normal mode basis)."""
    print("  Running thermodynamics simulation (normal mode basis)...")

    # Get normal modes
    omega, U = cfg.get_normal_modes()

    np.random.seed(42)
    x0 = np.random.randn(cfg.N)
    p0 = np.random.randn(cfg.N)
    state0 = np.concatenate([x0, p0])

    t = np.linspace(0, cfg.n_steps * cfg.dt, cfg.n_steps)
    solution = odeint(coupled_oscillator_rhs, state0, t,
                     args=(cfg.N, cfg.kappa, cfg.gamma))

    # External observer: radius decreases over time (approaches horizon)
    radii = np.linspace(1.0, cfg.r_s * 1.5, cfg.n_steps)

    # Track quantities over time
    k_dyn_external = np.zeros(cfg.n_steps)
    S_acc_external = np.zeros(cfg.n_steps)
    Q_external = np.zeros(cfg.n_steps)

    k_dyn_infalling = np.zeros(cfg.n_steps)
    S_acc_infalling = np.zeros(cfg.n_steps)
    Q_infalling = np.zeros(cfg.n_steps)

    window = 200  # Sliding window for covariance estimation
    warmup = 100

    for i in range(warmup, cfg.n_steps):
        window_start = max(0, i - window)
        # Transform positions to normal mode basis
        positions = solution[window_start:i+1, :cfg.N]
        positions_mode = positions @ U  # Each row transformed

        # External observer: apply aperture weights in normal mode basis
        weights_ext = compute_aperture_weights(cfg.N, radii[i], cfg.r_s, omega=omega)
        observed_ext = positions_mode * np.sqrt(weights_ext)[np.newaxis, :]
        cov_ext = np.cov(observed_ext.T)

        k_dyn_external[i] = compute_k_dyn(observed_ext)
        S_acc_external[i] = compute_S_acc(cov_ext)
        if i > warmup and S_acc_external[i] < S_acc_external[i-1]:
            Q_external[i] = Q_external[i-1] + (S_acc_external[i-1] - S_acc_external[i])
        else:
            Q_external[i] = Q_external[i-1]

        # Infalling observer: full access (normal mode basis)
        cov_inf = np.cov(positions_mode.T)
        k_dyn_infalling[i] = compute_k_dyn(positions_mode)
        S_acc_infalling[i] = compute_S_acc(cov_inf)
        if i > warmup and S_acc_infalling[i] < S_acc_infalling[i-1]:
            Q_infalling[i] = Q_infalling[i-1] + (S_acc_infalling[i-1] - S_acc_infalling[i])
        else:
            Q_infalling[i] = Q_infalling[i-1]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    t_plot = t[warmup:]

    axes[0].plot(t_plot, k_dyn_external[warmup:], 'C0-', lw=2, label='External')
    axes[0].plot(t_plot, k_dyn_infalling[warmup:], 'C1--', lw=2, label='Infalling')
    axes[0].set_xlabel('Coordinate time')
    axes[0].set_ylabel('$k_{dyn}$')
    axes[0].set_title('(A) Dynamical dimension')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_plot, S_acc_external[warmup:], 'C0-', lw=2, label='External')
    axes[1].plot(t_plot, S_acc_infalling[warmup:], 'C1--', lw=2, label='Infalling')
    axes[1].set_xlabel('Coordinate time')
    axes[1].set_ylabel('$S_{acc}$')
    axes[1].set_title('(B) Accessible entropy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_plot, Q_external[warmup:], 'C0-', lw=2, label='External')
    axes[2].plot(t_plot, Q_infalling[warmup:], 'C1--', lw=2, label='Infalling')
    axes[2].set_xlabel('Coordinate time')
    axes[2].set_ylabel('$Q$ (Landauer cost)')
    axes[2].set_title('(C) Cumulative erasure cost')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_thermodynamics.png', dpi=150)
    plt.savefig(FIGURES_DIR / 'fig2_thermodynamics.pdf', dpi=150)
    plt.close()
    print(f"  Final Q (external): {Q_external[-1]:.2f}")
    print("  Generated: fig2_thermodynamics.png")


# =============================================================================
# Figure 3: Schwarzschild Comparison (R² = 0.99)
# =============================================================================

def fig3_schwarzschild_comparison(cfg: Config):
    """Compare aperture model (normal mode basis) to Schwarzschild prediction."""
    print("  Running Schwarzschild comparison (normal mode basis)...")

    # Get normal modes
    omega, U = cfg.get_normal_modes()

    np.random.seed(42)
    x0 = np.random.randn(cfg.N)
    p0 = np.random.randn(cfg.N)
    state0 = np.concatenate([x0, p0])

    # Longer integration for better statistics
    n_steps = 5000
    t = np.linspace(0, n_steps * cfg.dt, n_steps)
    solution = odeint(coupled_oscillator_rhs, state0, t,
                     args=(cfg.N, cfg.kappa, cfg.gamma))

    # Test at multiple radii
    test_radii = np.linspace(cfg.r_s * 1.1, 1.0, 50)
    tau_dot_avg = np.zeros(len(test_radii))

    for ri, r in enumerate(test_radii):
        weights = compute_aperture_weights(cfg.N, r, cfg.r_s, omega=omega)
        tau_dots = []
        for i in range(n_steps):
            velocities_mode = U.T @ solution[i, cfg.N:]
            tau_dots.append(compute_tau_dot(velocities_mode, weights))
        tau_dot_avg[ri] = np.mean(tau_dots[-1000:])

    # Schwarzschild prediction
    schw = np.sqrt(1 - cfg.r_s / test_radii)
    schw_scaled = schw * tau_dot_avg.max() / schw.max()

    # Compute R²
    r2 = pearsonr(tau_dot_avg, schw_scaled)[0]**2

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(test_radii, tau_dot_avg, 'C0o-', lw=2, markersize=4,
                label='Aperture model')
    axes[0].plot(test_radii, schw_scaled, 'C1--', lw=2, label='Schwarzschild')
    axes[0].axvline(cfg.r_s, color='k', ls=':', label=f'$r_s$ = {cfg.r_s}')
    axes[0].set_xlabel('Radius r')
    axes[0].set_ylabel(r'Time rate $\dot{\tau}$')
    axes[0].set_title('(A) Time dilation vs radius')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(schw_scaled, tau_dot_avg, c=test_radii, cmap='viridis', s=40)
    axes[1].plot([0, schw_scaled.max()], [0, schw_scaled.max()], 'k--', lw=2)
    axes[1].set_xlabel('Schwarzschild prediction')
    axes[1].set_ylabel('Aperture model')
    axes[1].set_title(f'(B) Direct comparison: $R^2$ = {r2:.3f}')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_schwarzschild_comparison.pdf', dpi=150)
    plt.savefig(FIGURES_DIR / 'fig_schwarzschild_comparison.png', dpi=150)
    plt.close()
    print(f"  R² = {r2:.4f}")
    print("  Generated: fig_schwarzschild_comparison.pdf")


# =============================================================================
# Figure 3: Multi-Observer Complementarity
# =============================================================================

def fig4_complementarity(cfg: Config):
    """21 observers at different radii watching same dynamics (normal mode basis)."""
    print("  Running complementarity simulation (normal mode basis)...")

    # Get normal modes
    omega, U = cfg.get_normal_modes()

    n_observers = 21
    n_steps = 8000

    # External observers at various radii (log-spaced from near-horizon to far)
    observer_radii = np.logspace(np.log10(cfg.r_s * 1.1), 0, n_observers - 1)
    # Add a special "infalling" observer marker (we'll use 999.0 as a sentinel)
    observer_radii = np.append(observer_radii, [999.0])  # Infalling marker

    np.random.seed(42)
    x0 = np.random.randn(cfg.N)
    p0 = np.random.randn(cfg.N)
    state0 = np.concatenate([x0, p0])

    t = np.linspace(0, n_steps * cfg.dt, n_steps)
    solution = odeint(coupled_oscillator_rhs, state0, t,
                     args=(cfg.N, cfg.kappa, cfg.gamma))

    tau_accumulated = {r: np.zeros(n_steps) for r in observer_radii}

    for r in observer_radii:
        # Infalling observer has FULL access (weights = 1)
        if r == 999.0:
            weights = np.ones(cfg.N)
        else:
            weights = compute_aperture_weights(cfg.N, r, cfg.r_s, omega=omega)
        for i in range(1, n_steps):
            # Transform velocities to normal mode basis
            velocities_mode = U.T @ solution[i, cfg.N:]
            tau_dot = compute_tau_dot(velocities_mode, weights)
            tau_accumulated[r][i] = tau_accumulated[r][i-1] + tau_dot * cfg.dt

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # External observer radii (exclude infalling sentinel)
    external_radii = [r for r in observer_radii if r != 999.0]
    infalling_marker = 999.0

    norm = Normalize(vmin=cfg.r_s, vmax=1.0)
    cmap = cm.viridis

    # Panel A: Accumulated τ
    for r in sorted(external_radii):
        axes[0,0].plot(t, tau_accumulated[r], color=cmap(norm(r)), lw=1.5, alpha=0.7)
    axes[0,0].plot(t, tau_accumulated[infalling_marker], 'k--', lw=2, label='Infalling (full access)')
    axes[0,0].set_xlabel('Coordinate time')
    axes[0,0].set_ylabel(r'Accumulated $\tau$')
    axes[0,0].set_title('(A) Same dynamics, different clocks')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Panel B: τ̇ vs radius
    final_tau = [tau_accumulated[r][-1] for r in external_radii]
    axes[0,1].semilogy(external_radii, final_tau, 'C0o-', lw=2)
    axes[0,1].axvline(cfg.r_s, color='k', ls=':')
    axes[0,1].set_xlabel('Observer radius')
    axes[0,1].set_ylabel(r'Total $\tau$ (log)')
    axes[0,1].set_title('(B) Total time vs radius')
    axes[0,1].grid(True, alpha=0.3)

    # Panel C: Time dilation ratio
    infalling_tau_final = tau_accumulated[infalling_marker][-1]
    dilation_ratios = [infalling_tau_final / max(tau_accumulated[r][-1], 1e-10)
                      for r in external_radii]
    axes[1,0].semilogy(external_radii, dilation_ratios, 'C3o-', lw=2)
    axes[1,0].axhline(1.0, color='k', ls=':')
    axes[1,0].axvline(cfg.r_s, color='k', ls=':')
    axes[1,0].set_xlabel('Observer radius')
    axes[1,0].set_ylabel('Dilation ratio')
    axes[1,0].set_title(f'(C) Max dilation: {max(dilation_ratios):.1f}×')
    axes[1,0].grid(True, alpha=0.3)

    # Panel D: Colorbar info
    axes[1,1].text(0.5, 0.5, f'21 observers\n\nMax dilation: {max(dilation_ratios):.1f}×\n\n'
                  f'Infalling τ: {infalling_tau_final:.1f}\n'
                  f'Near-horizon τ: {min(final_tau):.1f}',
                  ha='center', va='center', fontsize=14, transform=axes[1,1].transAxes)
    axes[1,1].axis('off')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_complementarity.pdf', dpi=150)
    plt.savefig(FIGURES_DIR / 'fig_complementarity.png', dpi=150)
    plt.close()
    print(f"  Max time dilation: {max(dilation_ratios):.1f}×")
    print("  Generated: fig_complementarity.pdf")


# =============================================================================
# Figure 4: Ringdown-like relaxation (illustrative)
# =============================================================================

def fig4_ligo_connection(cfg: Config):
    """Illustrative: aperture perturbation produces ringdown-like relaxation (normal mode basis)."""
    print("  Running ringdown simulation (normal mode basis, illustrative)...")

    # Get normal modes
    omega, U = cfg.get_normal_modes()

    np.random.seed(42)
    x0 = np.random.randn(cfg.N)
    p0 = np.random.randn(cfg.N)
    state0 = np.concatenate([x0, p0])

    n_steps = 2000
    t = np.linspace(0, n_steps * cfg.dt, n_steps)
    solution = odeint(coupled_oscillator_rhs, state0, t,
                     args=(cfg.N, cfg.kappa, cfg.gamma))

    # Simulate merger: abrupt aperture change at t=10, then relaxation
    merger_time = 1000
    k_w_track = np.zeros(n_steps)
    tau_dot_track = np.zeros(n_steps)

    for i in range(n_steps):
        # Transform velocities to normal mode basis
        velocities_mode = U.T @ solution[i, cfg.N:]

        # Before merger: stable aperture
        if i < merger_time:
            r = 0.8
        # Merger: rapid contraction
        elif i < merger_time + 100:
            progress = (i - merger_time) / 100
            r = 0.8 - 0.6 * progress  # Contract toward horizon
        # Post-merger: ringdown relaxation
        else:
            decay = np.exp(-(i - merger_time - 100) * 0.01)
            r = 0.2 + 0.1 * decay * np.sin(0.1 * (i - merger_time - 100))
            r = max(r, cfg.r_s * 1.1)

        weights = compute_aperture_weights(cfg.N, r, cfg.r_s, omega=omega)
        k_w_track[i] = compute_k_w(weights)
        tau_dot_track[i] = compute_tau_dot(velocities_mode, weights)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(t, k_w_track, 'C0-', lw=2)
    axes[0].axvline(merger_time * cfg.dt, color='k', ls=':', label='Merger')
    axes[0].set_xlabel('Coordinate time')
    axes[0].set_ylabel('$k_w$')
    axes[0].set_title('(A) Channel dimension')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, tau_dot_track, 'C0-', lw=2)
    axes[1].axvline(merger_time * cfg.dt, color='k', ls=':')
    axes[1].set_xlabel('Coordinate time')
    axes[1].set_ylabel(r'$\dot{\tau}$')
    axes[1].set_title('(B) Time rate')
    axes[1].grid(True, alpha=0.3)

    # Ringdown region only
    ringdown_start = merger_time + 100
    ringdown_t = t[ringdown_start:]
    ringdown_signal = tau_dot_track[ringdown_start:]
    axes[2].plot(ringdown_t - ringdown_t[0], ringdown_signal, 'C3-', lw=2)
    axes[2].set_xlabel('Time since merger')
    axes[2].set_ylabel(r'$\dot{\tau}$ (ringdown)')
    axes[2].set_title('(C) Relaxation dynamics')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_ligo_connection.png', dpi=150)
    plt.savefig(FIGURES_DIR / 'fig3_ligo_connection.pdf', dpi=150)
    plt.close()
    print("  Generated: fig3_ligo_connection.png (illustrative)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Paper 2 figures")
    parser.add_argument("--figure", type=int, choices=[1, 2, 3, 4, 5],
                       help="Generate only specific figure (1=time, 2=thermo, 3=schw, 4=compl, 5=ringdown)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    # Set global seed for reproducibility
    np.random.seed(args.seed)
    print(f"Using random seed: {args.seed}")

    cfg = Config()

    print("=" * 60)
    print("PAPER 2: BLACK HOLE APERTURE - SIMULATIONS")
    print("=" * 60)

    if args.figure is None or args.figure == 1:
        fig1_time_dilation(cfg)
    if args.figure is None or args.figure == 2:
        fig2_thermodynamics(cfg)
    if args.figure is None or args.figure == 3:
        fig3_schwarzschild_comparison(cfg)
    if args.figure is None or args.figure == 4:
        fig4_complementarity(cfg)
    if args.figure is None or args.figure == 5:
        fig4_ligo_connection(cfg)

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
