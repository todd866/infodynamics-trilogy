#!/usr/bin/env python3
"""
DESI Dark Energy Prediction for Paper 3

The relaxation framework predicts specific w(z) evolution that can be
compared to DESI and Euclid data.

Key predictions:
1. w(z) is NOT constant (-1)
2. w(z) follows logistic relaxation: slow-fast-slow
3. w should be slightly > -1 around z ~ 1, approaching -1 at high/low z
4. The transition redshift depends on relaxation timescale

This outputs w(z) in a format directly comparable to DESI DR1.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
from scipy.interpolate import interp1d
from dataclasses import dataclass
import os

os.makedirs('../figures', exist_ok=True)


@dataclass
class RelaxationModel:
    """Parameters for the cosmic relaxation model."""
    s_0: float = 0.5          # Midpoint of relaxation (in substrate time)
    k: float = 3.0            # Relaxation rate constant
    R_max: float = 1.0        # Maximum relaxation
    w_inf: float = -1.0       # Asymptotic equation of state (de Sitter)
    delta_w_max: float = 0.15 # Maximum departure from w = -1
    z_transition: float = 0.8 # Transition redshift (where relaxation peaks)


def relaxation_rate(s: float, model: RelaxationModel) -> float:
    """
    Relaxation rate dR/ds = k * R * (1 - R/R_max)

    This is logistic: slow at start (rigid), fast in middle (loosening),
    slow at end (equilibrating).
    """
    R = model.R_max / (1 + np.exp(-model.k * (s - model.s_0)))
    dR_ds = model.k * R * (1 - R / model.R_max)
    return dR_ds


def w_from_relaxation(z: float, model: RelaxationModel) -> float:
    """
    Equation of state from relaxation dynamics.

    w(z) = w_∞ + δw(z)

    where δw peaks at the transition redshift.
    """
    # Map redshift to substrate time (approximate)
    # Higher z = earlier time = less relaxation
    # Use: s ∝ -log(1+z) as proxy
    s = -np.log(1 + z) / np.log(1 + model.z_transition) * model.s_0

    # Relaxation rate determines departure from w = -1
    rate = relaxation_rate(s, model)
    rate_max = model.k * model.R_max / 4  # Maximum of logistic rate

    # δw proportional to relaxation rate
    delta_w = model.delta_w_max * (rate / rate_max)

    return model.w_inf + delta_w


def compute_w_z(z_array: np.ndarray, model: RelaxationModel) -> np.ndarray:
    """Compute w(z) for array of redshifts."""
    return np.array([w_from_relaxation(z, model) for z in z_array])


def compute_hubble_evolution(z_array: np.ndarray, w_z: np.ndarray,
                             Omega_m: float = 0.3, H0: float = 70.0) -> np.ndarray:
    """
    Compute H(z)/H0 given w(z).

    H²(z)/H₀² = Ω_m(1+z)³ + Ω_DE(z)

    where Ω_DE(z) = Ω_DE,0 * exp(3 ∫₀ᶻ (1+w(z'))/(1+z') dz')
    """
    Omega_DE_0 = 1 - Omega_m

    # Interpolate w(z) for integration
    w_interp = interp1d(z_array, w_z, kind='linear', fill_value='extrapolate')

    def integrand(z):
        return (1 + w_interp(z)) / (1 + z)

    H_ratio = np.zeros_like(z_array)
    for i, z in enumerate(z_array):
        if z == 0:
            Omega_DE = Omega_DE_0
        else:
            integral, _ = quad(integrand, 0, z)
            Omega_DE = Omega_DE_0 * np.exp(3 * integral)

        H_ratio[i] = np.sqrt(Omega_m * (1 + z)**3 + Omega_DE)

    return H_ratio


def generate_desi_comparison(model: RelaxationModel):
    """
    Generate predictions in DESI-comparable format.
    """
    # DESI DR1 redshift bins (approximate)
    z_desi = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 2.5])

    # Fine grid for smooth curves
    z_fine = np.linspace(0, 3, 500)

    # Compute w(z) for both
    w_desi = compute_w_z(z_desi, model)
    w_fine = compute_w_z(z_fine, model)

    # ΛCDM comparison
    w_lcdm = np.ones_like(z_fine) * (-1.0)

    # Hubble evolution
    H_relaxation = compute_hubble_evolution(z_fine, w_fine)
    H_lcdm = compute_hubble_evolution(z_fine, w_lcdm)

    return {
        'z_desi': z_desi,
        'w_desi': w_desi,
        'z_fine': z_fine,
        'w_fine': w_fine,
        'w_lcdm': w_lcdm,
        'H_relaxation': H_relaxation,
        'H_lcdm': H_lcdm
    }


def plot_desi_prediction(data: dict, model: RelaxationModel):
    """Main figure showing DESI-testable predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    z_fine = data['z_fine']
    w_fine = data['w_fine']
    w_lcdm = data['w_lcdm']
    z_desi = data['z_desi']
    w_desi = data['w_desi']

    # --- Panel A: w(z) prediction ---
    ax1 = axes[0, 0]
    ax1.plot(z_fine, w_fine, 'C0-', lw=2.5, label='Relaxation model')
    ax1.plot(z_fine, w_lcdm, 'k--', lw=2, label=r'$\Lambda$CDM ($w = -1$)')
    ax1.scatter(z_desi, w_desi, c='C0', s=80, zorder=5, edgecolor='white',
                label='Prediction at DESI bins')

    # Shade the detectable region
    ax1.fill_between(z_fine, -1.05, -0.95, alpha=0.2, color='gray',
                     label='Current uncertainty')

    ax1.axvline(model.z_transition, color='C1', ls=':', alpha=0.7,
                label=f'Transition z = {model.z_transition}')

    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Equation of state w(z)')
    ax1.set_title('(A) Dark energy equation of state')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(-1.1, -0.8)
    ax1.grid(True, alpha=0.3)

    # --- Panel B: Departure from ΛCDM ---
    ax2 = axes[0, 1]
    delta_w = w_fine - (-1.0)
    ax2.plot(z_fine, delta_w * 100, 'C0-', lw=2.5)
    ax2.axhline(0, color='k', ls='--', lw=1)

    # DESI sensitivity (approximate)
    ax2.fill_between(z_fine, -2, 2, alpha=0.2, color='C1',
                     label='DESI ~2% sensitivity')

    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel(r'$\Delta w$ = $w(z) - (-1)$ [%]')
    ax2.set_title('(B) Departure from cosmological constant')
    ax2.legend()
    ax2.set_xlim(0, 3)
    ax2.set_ylim(-5, 20)
    ax2.grid(True, alpha=0.3)

    # Annotate maximum departure
    max_idx = np.argmax(delta_w)
    ax2.annotate(f'Max: {delta_w[max_idx]*100:.1f}%\nat z={z_fine[max_idx]:.1f}',
                 xy=(z_fine[max_idx], delta_w[max_idx]*100),
                 xytext=(z_fine[max_idx]+0.5, delta_w[max_idx]*100+3),
                 arrowprops=dict(arrowstyle='->', color='C0'),
                 fontsize=10)

    # --- Panel C: Hubble parameter ---
    ax3 = axes[1, 0]
    ax3.plot(z_fine, data['H_relaxation'], 'C0-', lw=2.5, label='Relaxation')
    ax3.plot(z_fine, data['H_lcdm'], 'k--', lw=2, label=r'$\Lambda$CDM')

    ax3.set_xlabel('Redshift z')
    ax3.set_ylabel(r'$H(z)/H_0$')
    ax3.set_title('(C) Hubble parameter evolution')
    ax3.legend()
    ax3.set_xlim(0, 3)
    ax3.grid(True, alpha=0.3)

    # --- Panel D: Fractional difference ---
    ax4 = axes[1, 1]
    H_diff = (data['H_relaxation'] - data['H_lcdm']) / data['H_lcdm'] * 100
    ax4.plot(z_fine, H_diff, 'C2-', lw=2.5)
    ax4.axhline(0, color='k', ls='--', lw=1)

    # BAO sensitivity
    ax4.fill_between(z_fine, -1, 1, alpha=0.2, color='C3',
                     label='BAO ~1% precision')

    ax4.set_xlabel('Redshift z')
    ax4.set_ylabel(r'$(H_{relax} - H_{\Lambda CDM})/H_{\Lambda CDM}$ [%]')
    ax4.set_title('(D) Detectable deviation in H(z)')
    ax4.legend()
    ax4.set_xlim(0, 3)
    ax4.set_ylim(-3, 5)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/fig_desi_prediction.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('../figures/fig_desi_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Output predictions in tabular form
    print("\n" + "=" * 60)
    print("DESI-COMPARABLE PREDICTIONS")
    print("=" * 60)
    print(f"{'z':>6} {'w(z)':>10} {'Δw [%]':>10}")
    print("-" * 30)
    for z, w in zip(z_desi, w_desi):
        print(f"{z:>6.2f} {w:>10.4f} {(w+1)*100:>10.2f}")
    print("=" * 60)
    print(f"Maximum departure: {(w_fine.max()+1)*100:.1f}% at z = {z_fine[np.argmax(w_fine)]:.2f}")
    print(f"Saved: fig_desi_prediction.pdf")


if __name__ == "__main__":
    model = RelaxationModel()

    print("=" * 60)
    print("DESI DARK ENERGY PREDICTION")
    print("=" * 60)
    print(f"  Transition redshift: z = {model.z_transition}")
    print(f"  Maximum w departure: Δw = {model.delta_w_max}")
    print("=" * 60)

    data = generate_desi_comparison(model)
    plot_desi_prediction(data, model)

    print("\nDone!")
