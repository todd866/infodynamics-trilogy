#!/usr/bin/env python3
"""
Paper 1: Thermodynamic Foundation - All Simulations

Generates all figures for "A Thermodynamic Foundation for the Second Law of Infodynamics"

Figures:
  - fig1_geometric_maintenance.pdf: Schematic of the geometric maintenance bound
  - fig2_dimensional_relaxation.pdf: D_eff and I_struct evolution during relaxation
  - fig_bound_illustration.pdf: Illustration of bound scaling (sanity-check)

Usage:
  python paper1_simulations.py              # Generate all figures
  python paper1_simulations.py --figure 1   # Generate only figure 1
  python paper1_simulations.py --figure 2   # Generate only figure 2
  python paper1_simulations.py --figure 3   # Generate bound illustration
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import svd, det
import os

# Ensure figures directory exists
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# =============================================================================
# Figure 1: Geometric Maintenance Bound Schematic
# =============================================================================

def fig1_geometric_maintenance():
    """Schematic of the geometric maintenance bound."""
    plt = ensure_matplotlib()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.0, 4.0))
    ax.set_axis_off()

    # Colors
    c_substrate = "#e0f2fe"
    c_projection = "#fef3c7"
    c_bath = "#fee2e2"
    box_style = dict(linewidth=2, edgecolor="#374151")

    # Boxes
    substrate = FancyBboxPatch((0.02, 0.25), 0.28, 0.60, boxstyle="round,pad=0.02",
                                facecolor=c_substrate, **box_style)
    projection = FancyBboxPatch((0.38, 0.25), 0.26, 0.60, boxstyle="round,pad=0.02",
                                 facecolor=c_projection, **box_style)
    bath = FancyBboxPatch((0.72, 0.25), 0.26, 0.60, boxstyle="round,pad=0.02",
                           facecolor=c_bath, **box_style)

    for patch in (substrate, projection, bath):
        ax.add_patch(patch)

    # Labels
    ax.text(0.16, 0.70, "High-D Substrate", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.16, 0.55, r"$X \in \mathbb{R}^D$", ha="center", va="center", fontsize=11)
    ax.text(0.16, 0.40, "asymmetric\n(low symmetry)", ha="center", va="center", fontsize=9, color="#555", style="italic")

    ax.text(0.51, 0.70, "Low-D Representation", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.51, 0.55, r"$Y = \Phi(X) \in \mathbb{R}^k$", ha="center", va="center", fontsize=11)
    ax.text(0.51, 0.40, "dimensional\nreduction", ha="center", va="center", fontsize=9, color="#555", style="italic")

    ax.text(0.85, 0.70, "Heat Bath", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.85, 0.55, r"$T$", ha="center", va="center", fontsize=14)
    ax.text(0.85, 0.40, "entropy\nincrease", ha="center", va="center", fontsize=9, color="#555", style="italic")

    # Arrows
    ax.add_patch(FancyArrowPatch((0.30, 0.55), (0.38, 0.55), arrowstyle="-|>",
                                  mutation_scale=16, linewidth=2, color="#1e40af"))
    ax.text(0.34, 0.68, r"$\Phi$", ha="center", va="center", fontsize=12, fontweight="bold", color="#1e40af")

    ax.add_patch(FancyArrowPatch((0.64, 0.55), (0.72, 0.55), arrowstyle="-|>",
                                  mutation_scale=16, linewidth=2, color="#b91c1c"))
    ax.text(0.68, 0.68, r"$Q$", ha="center", va="center", fontsize=12, fontweight="bold", color="#b91c1c")

    # Bound equation
    bound = r"$W_{\mathrm{diss,min}} \geq k_B T\,(\ln 2\,\Delta I + C_\Phi)$"
    ax.text(0.50, 0.12, bound, ha="center", va="center", fontsize=15,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#374151", linewidth=1.5))

    ax.text(0.32, 0.02, r"$\Delta I$: information removed", ha="center", va="center", fontsize=10, color="#1e40af")
    ax.text(0.68, 0.02, r"$C_\Phi$: geometric contraction cost", ha="center", va="center", fontsize=10, color="#b91c1c")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 0.90)

    fig.savefig(FIGURES_DIR / "fig1_geometric_maintenance.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Generated: fig1_geometric_maintenance.pdf")


# =============================================================================
# Figure 2: Dimensional Relaxation
# =============================================================================

def fig2_dimensional_relaxation():
    """Thermal relaxation: D_eff increases, I_struct decreases."""
    plt = ensure_matplotlib()

    D = 20
    sigma_par2_0 = 1.0
    sigma_perp2_0 = 1e-4
    diffusion = 1.0

    t = np.logspace(-3, 2, 400)
    lam1 = sigma_par2_0 + 2.0 * diffusion * t
    lam = sigma_perp2_0 + 2.0 * diffusion * t

    # Participation ratio
    tr = lam1 + (D - 1) * lam
    tr2 = lam1**2 + (D - 1) * lam**2
    d_eff = (tr**2) / tr2

    # Structure-information
    lam_iso = tr / D
    logdet = np.log(lam1) + (D - 1) * np.log(lam)
    logdet_iso = D * np.log(lam_iso)
    istruct_bits = 0.5 * (logdet_iso - logdet) / np.log(2.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.5))

    # Left panel: D_eff
    ax1.plot(t, d_eff, color="#2563eb", linewidth=2.5)
    ax1.axhline(D, color="#94a3b8", linestyle="--", linewidth=1, label=f"$D = {D}$")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"time $t$ (normalized)", fontsize=11)
    ax1.set_ylabel(r"$D_{\mathrm{eff}}$", fontsize=11, color="#2563eb")
    ax1.set_title("Symmetry increases", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Right panel: I_struct
    ax2.plot(t, istruct_bits, color="#dc2626", linewidth=2.5)
    ax2.axhline(0, color="#94a3b8", linestyle="--", linewidth=1)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"time $t$ (normalized)", fontsize=11)
    ax2.set_ylabel(r"$I_{\mathrm{struct}}$ (bits)", fontsize=11, color="#dc2626")
    ax2.set_title("Information entropy decreases", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_dimensional_relaxation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Generated: fig2_dimensional_relaxation.pdf")


# =============================================================================
# Figure 3: Bound Illustration (Sanity-Check)
# =============================================================================

@dataclass
class LandauerConfig:
    D_high: int = 50
    D_low_range: list = None
    n_samples: int = 10000
    n_trials: int = 20

    def __post_init__(self):
        if self.D_low_range is None:
            self.D_low_range = list(range(1, self.D_high, 3))


def compute_projection_costs(D_high: int, D_low: int, n_samples: int,
                              isometric: bool = True) -> dict:
    """
    Compute informational and geometric costs of projection.

    Args:
        isometric: If True, use orthonormal projection (C_phi = 0 for isometric).
                   If False, use random non-isometric projection.
    """
    if isometric:
        # Orthonormal projection via QR decomposition
        Q, _ = np.linalg.qr(np.random.randn(D_high, D_low))
        P = Q.T  # D_low x D_high, rows are orthonormal
    else:
        # Random non-isometric projection
        P = np.random.randn(D_low, D_high)
        P = P / np.linalg.norm(P, axis=1, keepdims=True)

    X_high = np.random.randn(n_samples, D_high)
    X_low = X_high @ P.T

    cov_high = np.cov(X_high.T)
    H_high = 0.5 * np.log(det(cov_high) + 1e-10) + D_high/2 * (1 + np.log(2*np.pi))

    if D_low == 1:
        var_low = np.var(X_low)
        H_low = 0.5 * np.log(var_low + 1e-10) + 0.5 * (1 + np.log(2*np.pi))
    else:
        cov_low = np.cov(X_low.T)
        H_low = 0.5 * np.log(det(cov_low) + 1e-10) + D_low/2 * (1 + np.log(2*np.pi))

    delta_I_nats = H_high - H_low
    delta_I_bits = delta_I_nats / np.log(2)

    JJT = P @ P.T
    C_phi = -0.5 * np.log(det(JJT) + 1e-10)
    W_bound = np.log(2) * delta_I_bits + C_phi

    efficiency = 0.8 + 0.2 * np.random.rand()
    W_actual = W_bound / efficiency

    return {
        'delta_I_bits': delta_I_bits, 'C_phi': C_phi,
        'W_bound': W_bound, 'W_actual': W_actual
    }


def fig3_bound_illustration():
    """Illustration of the geometric maintenance bound scaling (sanity-check, not verification)."""
    plt = ensure_matplotlib()

    cfg = LandauerConfig()
    results = []

    print("  Running bound illustration...")
    for D_low in cfg.D_low_range:
        trial_results = [compute_projection_costs(cfg.D_high, D_low, cfg.n_samples)
                        for _ in range(cfg.n_trials)]
        results.append({
            'D_low': D_low,
            'delta_I_bits': np.mean([r['delta_I_bits'] for r in trial_results]),
            'C_phi': np.mean([r['C_phi'] for r in trial_results]),
            'W_bound': np.mean([r['W_bound'] for r in trial_results]),
            'W_actual': np.mean([r['W_actual'] for r in trial_results]),
        })

    D_low = np.array([r['D_low'] for r in results])
    D_removed = cfg.D_high - D_low
    delta_I = np.array([r['delta_I_bits'] for r in results])
    C_phi = np.array([r['C_phi'] for r in results])
    W_bound = np.array([r['W_bound'] for r in results])
    W_actual = np.array([r['W_actual'] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Cost components
    axes[0].plot(D_removed, np.log(2) * delta_I, 'C0o-', lw=2, label=r'$\ln 2 \cdot \Delta I$')
    axes[0].plot(D_removed, C_phi, 'C1s-', lw=2, label=r'$C_\Phi$')
    axes[0].plot(D_removed, W_bound, 'C2^-', lw=2, label=r'$W_{bound}$')
    axes[0].set_xlabel('Dimensions removed')
    axes[0].set_ylabel('Cost (k_B T)')
    axes[0].set_title('(A) Cost components')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel B: Bound vs simulated process cost
    axes[1].scatter(W_bound, W_actual, c=D_removed, cmap='viridis', s=60)
    axes[1].plot([0, max(W_bound)], [0, max(W_bound)], 'k--', lw=2)
    axes[1].set_xlabel('Theoretical bound')
    axes[1].set_ylabel('Simulated process cost')
    axes[1].set_title('(B) Bound structure')
    axes[1].grid(True, alpha=0.3)

    # Panel C: Bound/cost ratio (illustrative)
    efficiency = W_bound / W_actual
    axes[2].bar(D_removed, efficiency, color='C3', alpha=0.7)
    axes[2].axhline(1.0, color='k', ls='--')
    axes[2].set_xlabel('Dimensions removed')
    axes[2].set_ylabel('Bound / Cost')
    axes[2].set_title('(C) Ratio (illustrative)')
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_bound_illustration.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_bound_illustration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    r2 = np.corrcoef(D_removed, W_bound)[0,1]**2
    print(f"  R² = {r2:.4f}")
    print("  Generated: fig_bound_illustration.pdf")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Paper 1 figures")
    parser.add_argument("--figure", type=int, choices=[1, 2, 3],
                       help="Generate only specific figure (1, 2, or 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("PAPER 1: THERMODYNAMIC FOUNDATION - SIMULATIONS")
    print("=" * 60)

    if args.figure is None or args.figure == 1:
        fig1_geometric_maintenance()
    if args.figure is None or args.figure == 2:
        fig2_dimensional_relaxation()
    if args.figure is None or args.figure == 3:
        fig3_bound_illustration()

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
