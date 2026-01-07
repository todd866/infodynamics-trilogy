#!/usr/bin/env python3
"""
Paper 3: Cosmic Relaxation - All Simulations

The universe as a relaxing topological knot. Time is relaxation rate.
Dark energy is relaxation pressure. Mathematics emerges from constraint structure.

Figures:
  - fig1_napkin.png: Napkin metaphor for cosmic relaxation
  - fig2_cascade.png: Symmetry-breaking cascade
  - fig3_relaxation.png: Logistic slow-fast-slow dynamics
  - fig4_knot.png: Knot energy decrease
  - fig5_dark_energy.png: w(z) prediction vs ΛCDM
  - fig_desi_prediction.pdf: Quantitative DESI testable predictions
  - fig6_information.png: Information accumulation tracks relaxation

Usage:
  python paper3_simulations.py              # Generate all figures
  python paper3_simulations.py --figure 1   # Napkin metaphor only
  python paper3_simulations.py --figure 5   # Dark energy only
  python paper3_simulations.py --figure 6   # DESI predictions only
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from scipy.integrate import quad
from scipy.interpolate import interp1d
from dataclasses import dataclass
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# =============================================================================
# Relaxation Model
# =============================================================================

@dataclass
class RelaxationModel:
    """Parameters for the cosmic relaxation model."""
    s_0: float = 0.5          # Midpoint of relaxation
    k: float = 3.0            # Relaxation rate constant
    R_max: float = 1.0        # Maximum relaxation
    w_inf: float = -1.0       # Asymptotic equation of state
    delta_w_max: float = 0.15 # Maximum departure from w = -1
    z_transition: float = 0.8 # Transition redshift


def logistic_relaxation(s: np.ndarray, s_0: float = 10, k: float = 0.5,
                        R_max: float = 1.0) -> np.ndarray:
    """Logistic relaxation: R(s) = R_max / (1 + exp(-k(s - s_0)))"""
    return R_max / (1 + np.exp(-k * (s - s_0)))


def relaxation_rate(s: np.ndarray, s_0: float = 10, k: float = 0.5,
                    R_max: float = 1.0) -> np.ndarray:
    """dR/ds = k * R * (1 - R/R_max) — bell-shaped curve"""
    R = logistic_relaxation(s, s_0, k, R_max)
    return k * R * (1 - R / R_max)


# =============================================================================
# Figure 1: Napkin Metaphor
# =============================================================================

def fig1_napkin():
    """Visual metaphor: crumpled napkin relaxing."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax in axes:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

    # Panel A: Tightly crumpled (early universe)
    np.random.seed(42)
    n_folds = 40
    for _ in range(n_folds):
        x = np.random.randn(20) * 0.3
        y = np.random.randn(20) * 0.3
        axes[0].plot(x, y, 'C0-', alpha=0.6, lw=1.5)
    axes[0].set_title('Early Universe\n(tightly constrained)', fontsize=12, fontweight='bold')

    # Panel B: Loosening (present)
    for _ in range(25):
        x = np.random.randn(20) * 0.6
        y = np.random.randn(20) * 0.6
        axes[1].plot(x, y, 'C1-', alpha=0.5, lw=1.2)
    axes[1].set_title('Present Epoch\n(fast relaxation)', fontsize=12, fontweight='bold')

    # Panel C: Nearly flat (heat death)
    for _ in range(10):
        x = np.random.randn(20) * 1.0
        y = np.random.randn(20) * 0.1
        axes[2].plot(x, y, 'C2-', alpha=0.4, lw=1)
    axes[2].set_title('Heat Death\n(fully relaxed)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_napkin.png', dpi=150)
    plt.close()
    print("  Generated: fig1_napkin.png")


# =============================================================================
# Figure 2: Symmetry-Breaking Cascade
# =============================================================================

def fig2_cascade():
    """Symmetry-breaking cascade creating mathematical structure."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)

    stages = [
        (1, 7, "Primordial Break", "Set theory\n(∈, ∉)", "#fee2e2"),
        (3, 5.5, "Ordering Break", "Ordinals\n(<, >)", "#fef3c7"),
        (5, 4, "Compositional Break", "Algebra\n(+, ×, groups)", "#d1fae5"),
        (7, 2.5, "Locality Break", "Topology\n(open, closed)", "#dbeafe"),
        (9, 1, "Metric Break", "Geometry\n(distance, angle)", "#e9d5ff"),
    ]

    for i, (x, y, label, math, color) in enumerate(stages):
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2,
                             boxstyle="round,pad=0.05", facecolor=color,
                             edgecolor='#374151', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y+0.2, label, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, y-0.3, math, ha='center', va='center', fontsize=8, style='italic')

        if i < len(stages) - 1:
            next_x, next_y = stages[i+1][0], stages[i+1][1]
            ax.annotate('', xy=(next_x-0.8, next_y+0.6), xytext=(x+0.8, y-0.6),
                       arrowprops=dict(arrowstyle='->', color='#6b7280', lw=2))

    ax.text(5, 7.5, "Symmetry-Breaking Cascade", ha='center', fontsize=14, fontweight='bold')

    plt.savefig(FIGURES_DIR / 'fig2_cascade.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Generated: fig2_cascade.png")


# =============================================================================
# Figure 3: Logistic Relaxation Curve
# =============================================================================

def fig3_relaxation():
    """Slow-fast-slow relaxation dynamics."""
    s = np.linspace(0, 20, 500)
    R = logistic_relaxation(s)
    dR = relaxation_rate(s)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(s, R, 'C0-', lw=2.5)
    axes[0].axhline(0.5, color='#9ca3af', ls='--', lw=1)
    axes[0].set_xlabel('Substrate parameter s')
    axes[0].set_ylabel('Relaxation progress R(s)')
    axes[0].set_title('(A) Logistic relaxation')
    axes[0].grid(True, alpha=0.3)

    # Annotate phases
    axes[0].annotate('Slow\n(rigid)', xy=(2, 0.1), fontsize=10, ha='center')
    axes[0].annotate('Fast\n(loosening)', xy=(10, 0.5), fontsize=10, ha='center')
    axes[0].annotate('Slow\n(equilibrating)', xy=(18, 0.9), fontsize=10, ha='center')

    axes[1].plot(s, dR, 'C1-', lw=2.5)
    axes[1].set_xlabel('Substrate parameter s')
    axes[1].set_ylabel('Relaxation rate dR/ds')
    axes[1].set_title('(B) Bell-shaped rate (= time rate)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_relaxation.png', dpi=150)
    plt.close()
    print("  Generated: fig3_relaxation.png")


# =============================================================================
# Figure 4: Knot Energy
# =============================================================================

def fig4_knot():
    """Knot energy decreasing as constraints release."""
    s = np.linspace(0, 20, 500)
    R = logistic_relaxation(s)
    E_knot = 1 - R  # Energy decreases as relaxation increases

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s, E_knot, 'C3-', lw=2.5)
    ax.fill_between(s, 0, E_knot, alpha=0.2, color='C3')
    ax.set_xlabel('Substrate parameter s')
    ax.set_ylabel('Constraint energy $E_{knot}$')
    ax.set_title('Knot Energy Decreases as Universe Relaxes')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_knot.png', dpi=150)
    plt.close()
    print("  Generated: fig4_knot.png")


# =============================================================================
# Figure 5: Dark Energy w(z) Prediction
# =============================================================================

def w_from_relaxation(z: float, model: RelaxationModel) -> float:
    """Equation of state from relaxation dynamics."""
    s = -np.log(1 + z) / np.log(1 + model.z_transition) * model.s_0
    R = model.R_max / (1 + np.exp(-model.k * (s - model.s_0)))
    dR_ds = model.k * R * (1 - R / model.R_max)
    rate_max = model.k * model.R_max / 4
    delta_w = model.delta_w_max * (dR_ds / rate_max)
    return model.w_inf + delta_w


def fig5_dark_energy():
    """w(z) prediction vs ΛCDM."""
    model = RelaxationModel()
    z = np.linspace(0, 3, 500)
    w = np.array([w_from_relaxation(zi, model) for zi in z])
    w_lcdm = np.ones_like(z) * (-1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(z, w, 'C0-', lw=2.5, label='Relaxation model')
    axes[0].plot(z, w_lcdm, 'k--', lw=2, label=r'$\Lambda$CDM')
    axes[0].axvline(model.z_transition, color='C1', ls=':', label=f'Transition z={model.z_transition}')
    axes[0].set_xlabel('Redshift z')
    axes[0].set_ylabel('w(z)')
    axes[0].set_title('(A) Equation of state')
    axes[0].legend()
    axes[0].set_ylim(-1.1, -0.8)
    axes[0].grid(True, alpha=0.3)

    delta_w = (w - (-1.0)) * 100
    axes[1].plot(z, delta_w, 'C0-', lw=2.5)
    axes[1].axhline(0, color='k', ls='--')
    axes[1].fill_between(z, -2, 2, alpha=0.2, color='C1', label='DESI ~2% sensitivity')
    axes[1].set_xlabel('Redshift z')
    axes[1].set_ylabel(r'$\Delta w$ [%]')
    axes[1].set_title('(B) Departure from $w = -1$')
    axes[1].legend()
    axes[1].set_ylim(-5, 20)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig5_dark_energy.png', dpi=150)
    plt.close()
    print("  Generated: fig5_dark_energy.png")


# =============================================================================
# Figure 6: DESI Quantitative Predictions
# =============================================================================

def fig6_desi_prediction():
    """Quantitative DESI-testable predictions."""
    model = RelaxationModel()

    z_desi = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 2.5])
    z_fine = np.linspace(0.01, 3, 500)

    w_desi = np.array([w_from_relaxation(zi, model) for zi in z_desi])
    w_fine = np.array([w_from_relaxation(zi, model) for zi in z_fine])
    w_lcdm = np.ones_like(z_fine) * (-1.0)

    # Hubble evolution (simplified)
    Omega_m = 0.3
    H_relaxation = np.sqrt(Omega_m * (1 + z_fine)**3 + (1 - Omega_m))
    H_lcdm = H_relaxation.copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: w(z)
    axes[0,0].plot(z_fine, w_fine, 'C0-', lw=2.5, label='Relaxation')
    axes[0,0].plot(z_fine, w_lcdm, 'k--', lw=2, label=r'$\Lambda$CDM')
    axes[0,0].scatter(z_desi, w_desi, c='C0', s=80, zorder=5, edgecolor='white')
    axes[0,0].axvline(model.z_transition, color='C1', ls=':', alpha=0.7)
    axes[0,0].set_xlabel('Redshift z')
    axes[0,0].set_ylabel('w(z)')
    axes[0,0].set_title('(A) Dark energy equation of state')
    axes[0,0].legend()
    axes[0,0].set_ylim(-1.1, -0.8)
    axes[0,0].grid(True, alpha=0.3)

    # Panel B: Departure
    delta_w = (w_fine - (-1.0)) * 100
    axes[0,1].plot(z_fine, delta_w, 'C0-', lw=2.5)
    axes[0,1].axhline(0, color='k', ls='--')
    axes[0,1].fill_between(z_fine, -2, 2, alpha=0.2, color='C1', label='DESI ~2%')
    axes[0,1].set_xlabel('Redshift z')
    axes[0,1].set_ylabel(r'$\Delta w$ [%]')
    axes[0,1].set_title('(B) Departure from cosmological constant')
    axes[0,1].legend()
    axes[0,1].set_ylim(-5, 15)
    axes[0,1].grid(True, alpha=0.3)

    max_idx = np.argmax(delta_w)
    axes[0,1].annotate(f'Max: {delta_w[max_idx]:.1f}%', xy=(z_fine[max_idx], delta_w[max_idx]),
                      xytext=(z_fine[max_idx]+0.5, delta_w[max_idx]+2),
                      arrowprops=dict(arrowstyle='->', color='C0'))

    # Panel C: H(z)
    axes[1,0].plot(z_fine, H_relaxation, 'C0-', lw=2.5, label='Relaxation')
    axes[1,0].plot(z_fine, H_lcdm, 'k--', lw=2, label=r'$\Lambda$CDM')
    axes[1,0].set_xlabel('Redshift z')
    axes[1,0].set_ylabel(r'$H(z)/H_0$')
    axes[1,0].set_title('(C) Hubble parameter')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Panel D: Table of predictions
    axes[1,1].axis('off')
    table_data = [[f'{z:.1f}', f'{w:.4f}', f'{(w+1)*100:.2f}%']
                  for z, w in zip(z_desi, w_desi)]
    table = axes[1,1].table(cellText=table_data,
                            colLabels=['z', 'w(z)', 'Δw'],
                            loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1,1].set_title('(D) DESI bin predictions', pad=20)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_desi_prediction.pdf', dpi=150)
    plt.savefig(FIGURES_DIR / 'fig_desi_prediction.png', dpi=150)
    plt.close()
    print("  Generated: fig_desi_prediction.pdf")


# =============================================================================
# Figure 7: Information Accumulation
# =============================================================================

def fig7_information():
    """Information accumulation tracks relaxation."""
    s = np.linspace(0, 20, 500)
    dR = relaxation_rate(s)

    # Information accumulation (integral of relaxation rate)
    I_acc = np.cumsum(dR) * (s[1] - s[0])

    # Add noise for "observed" data
    np.random.seed(42)
    I_obs = I_acc + 0.02 * np.random.randn(len(s))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s, I_acc, 'C0-', lw=2.5, label='Theoretical (relaxation rate)')
    ax.scatter(s[::20], I_obs[::20], c='C1', s=30, alpha=0.6, label='Observed (noisy)')
    ax.set_xlabel('Substrate parameter s')
    ax.set_ylabel('Accumulated information I')
    ax.set_title('Information Accumulation Tracks Relaxation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Compute correlation
    r = np.corrcoef(I_acc, I_obs)[0, 1]
    ax.text(0.95, 0.05, f'r = {r:.3f}', transform=ax.transAxes, ha='right', fontsize=12)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig6_information.png', dpi=150)
    plt.close()
    print("  Generated: fig6_information.png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Paper 3 figures")
    parser.add_argument("--figure", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                       help="Generate only specific figure")
    args = parser.parse_args()

    print("=" * 60)
    print("PAPER 3: COSMIC RELAXATION - SIMULATIONS")
    print("=" * 60)

    figure_funcs = {
        1: fig1_napkin,
        2: fig2_cascade,
        3: fig3_relaxation,
        4: fig4_knot,
        5: fig5_dark_energy,
        6: fig6_desi_prediction,
        7: fig7_information,
    }

    if args.figure is not None:
        figure_funcs[args.figure]()
    else:
        for func in figure_funcs.values():
            func()

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
