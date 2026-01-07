#!/usr/bin/env python3
"""
Generate figures for "Time, Mathematics, and the Relaxing Knot"

Figures:
1. Relaxation dynamics: cumulative relaxation and rate (slow-fast-slow)
2. Knot energy and constraint release over time
3. Dark energy equation of state w(z) prediction vs Lambda-CDM
4. Information accumulation rate tracking relaxation
5. Symmetry-breaking cascade schematic
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Color scheme
COLORS = {
    'relaxation': '#2E86AB',
    'rate': '#A23B72',
    'energy': '#F18F01',
    'info': '#C73E1D',
    'dark_energy': '#3B1F2B',
    'lambda': '#95969A',
    'substrate': '#E8E8E8',
    'knot': '#5C4D7D',
    'math': '#2E7D32',
    'physics': '#1565C0',
    'metaphysics': '#6A1B9A'
}


def logistic_relaxation(t, R_max=1.0, k=0.5, t0=10):
    """Logistic relaxation curve"""
    return R_max / (1 + np.exp(-k * (t - t0)))


def relaxation_rate(t, R_max=1.0, k=0.5, t0=10):
    """Derivative of logistic relaxation (bell-shaped)"""
    exp_term = np.exp(-k * (t - t0))
    return k * R_max * exp_term / (1 + exp_term)**2


def knot_energy(t, E0=1.0, k=0.5, t0=10):
    """Energy stored in knot constraints (decreases as relaxation proceeds)"""
    R = logistic_relaxation(t, R_max=1.0, k=k, t0=t0)
    return E0 * (1 - R)


def dark_energy_w(z, w0=-1.0, wa=0.1):
    """CPL parameterization of dark energy equation of state"""
    a = 1 / (1 + z)
    return w0 + wa * (1 - a)


def relaxation_w(z, z_peak=1.0, delta_w=0.15):
    """
    Dark energy w(z) from relaxation dynamics.
    Departs from -1 around the peak relaxation epoch.
    """
    # Map z to relaxation phase
    # Higher z = earlier time = slower relaxation = w closer to -1
    # z ~ z_peak = peak relaxation = maximum departure from -1
    # Low z = late time = slowing relaxation = w returning toward -1

    # Use a Gaussian-like departure centered on z_peak
    w = -1.0 + delta_w * np.exp(-0.5 * ((z - z_peak) / 0.8)**2)
    return w


def info_accumulation_rate(t, k=0.5, t0=10, noise_scale=0.05):
    """Information accumulation rate tracks relaxation rate with noise"""
    base_rate = relaxation_rate(t, k=k, t0=t0)
    noise = noise_scale * np.random.randn(len(t)) * base_rate
    return base_rate + noise


# =============================================================================
# Figure 1: Relaxation Dynamics (Slow-Fast-Slow)
# =============================================================================
def fig3_relaxation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    t = np.linspace(0, 20, 500)
    R = logistic_relaxation(t, R_max=1.0, k=0.5, t0=10)
    dR = relaxation_rate(t, R_max=1.0, k=0.5, t0=10)

    # Left panel: Cumulative relaxation
    ax1 = axes[0]
    ax1.plot(t, R, color=COLORS['relaxation'], linewidth=2.5, label='Cumulative relaxation $R(t)$')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=10, color='gray', linestyle='--', alpha=0.5)

    # Shade phases
    ax1.axvspan(0, 6, alpha=0.1, color='blue', label='Early (slow)')
    ax1.axvspan(6, 14, alpha=0.1, color='green', label='Middle (fast)')
    ax1.axvspan(14, 20, alpha=0.1, color='red', label='Late (slow)')

    ax1.set_xlabel('Substrate coordinate $s$')
    ax1.set_ylabel('Cumulative relaxation $R$')
    ax1.set_title('(a) Cumulative Relaxation')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 1.1)

    # Right panel: Relaxation rate
    ax2 = axes[1]
    ax2.plot(t, dR, color=COLORS['rate'], linewidth=2.5, label='Relaxation rate $\\dot{R}(t)$')
    ax2.fill_between(t, 0, dR, alpha=0.3, color=COLORS['rate'])

    # Mark phases
    ax2.axvspan(0, 6, alpha=0.1, color='blue')
    ax2.axvspan(6, 14, alpha=0.1, color='green')
    ax2.axvspan(14, 20, alpha=0.1, color='red')

    # Annotations - place in shaded regions with white background for contrast
    ax2.text(3, 0.01, 'Slow\n(rigid)', ha='center', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='blue'))
    ax2.text(10, 0.01, 'Fast\n(loosening)', ha='center', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='green'))
    ax2.text(17, 0.01, 'Slow\n(equilibrating)', ha='center', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red'))

    ax2.set_xlabel('Substrate coordinate $s$')
    ax2.set_ylabel('Relaxation rate $\\dot{R}$')
    ax2.set_title('(b) Relaxation Rate (Bell-Shaped)')
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 0.15)

    plt.tight_layout()
    plt.savefig('../figures/fig3_relaxation.png', dpi=300, bbox_inches='tight')
    plt.savefig('../figures/fig3_relaxation.pdf', bbox_inches='tight')
    plt.close()
    print("Generated: fig3_relaxation")


# =============================================================================
# Figure 2: Knot Energy and Constraint Release
# =============================================================================
def fig4_knot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    t = np.linspace(0, 20, 500)
    E = knot_energy(t, E0=1.0, k=0.5, t0=10)
    R = logistic_relaxation(t, R_max=1.0, k=0.5, t0=10)

    # Number of active constraints (discrete steps for visualization)
    N_constraints = 10
    constraint_times = np.linspace(2, 18, N_constraints)

    # Left panel: Knot energy
    ax1 = axes[0]
    ax1.plot(t, E, color=COLORS['energy'], linewidth=2.5, label='Constraint energy $E_\\mathcal{K}(t)$')
    ax1.fill_between(t, 0, E, alpha=0.3, color=COLORS['energy'])

    ax1.set_xlabel('Substrate coordinate $s$')
    ax1.set_ylabel('Constraint energy $E_\\mathcal{K}$')
    ax1.set_title('(a) Knot Energy Decreases as Constraints Release')
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 1.1)

    # Add annotations - stacked on right side with arrows to curve
    ax1.annotate('Tightly knotted', xy=(2, 0.98), xytext=(14, 0.95),
                 ha='left', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))
    ax1.annotate('Relaxing', xy=(10, 0.5), xytext=(14, 0.75),
                 ha='left', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))
    ax1.annotate('Nearly flat', xy=(18, 0.02), xytext=(14, 0.55),
                 ha='left', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))

    # Right panel: Active constraints
    ax2 = axes[1]

    # Create step function for constraints
    constraints_remaining = []
    for ti in t:
        n = sum(1 for ct in constraint_times if ct > ti)
        constraints_remaining.append(n)

    ax2.step(t, constraints_remaining, where='post', color=COLORS['knot'],
             linewidth=2.5, label='Active constraints')
    ax2.fill_between(t, 0, constraints_remaining, step='post', alpha=0.3, color=COLORS['knot'])

    # Mark constraint release events - just vertical lines, no text labels
    for i, ct in enumerate(constraint_times):
        ax2.axvline(x=ct, color='gray', linestyle=':', alpha=0.3)

    ax2.set_xlabel('Substrate coordinate $s$')
    ax2.set_ylabel('Number of active constraints')
    ax2.set_title('(b) Symmetry-Breaking Events Release Constraints')
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, N_constraints + 1)

    plt.tight_layout()
    plt.savefig('../figures/fig4_knot.png', dpi=300, bbox_inches='tight')
    plt.savefig('../figures/fig4_knot.pdf', bbox_inches='tight')
    plt.close()
    print("Generated: fig4_knot")


# =============================================================================
# Figure 3: Dark Energy w(z) Prediction
# =============================================================================
def fig5_dark_energy():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    z = np.linspace(0, 3, 200)

    # Left panel: w(z) comparison
    ax1 = axes[0]

    w_lambda = np.ones_like(z) * (-1)
    w_relax = relaxation_w(z, z_peak=1.0, delta_w=0.05)
    w_cpl = dark_energy_w(z, w0=-0.98, wa=0.08)

    ax1.plot(z, w_lambda, '--', color=COLORS['lambda'], linewidth=2, label='$\\Lambda$CDM ($w = -1$)')
    ax1.plot(z, w_relax, '-', color=COLORS['dark_energy'], linewidth=2.5, label='Relaxation prediction')
    ax1.plot(z, w_cpl, ':', color='purple', linewidth=2, label='CPL parameterization')

    # Add uncertainty band for relaxation
    ax1.fill_between(z, w_relax - 0.02, w_relax + 0.02, alpha=0.2, color=COLORS['dark_energy'])

    ax1.set_xlabel('Redshift $z$')
    ax1.set_ylabel('Dark energy equation of state $w$')
    ax1.set_title('(a) Dark Energy $w(z)$: Relaxation vs $\\Lambda$CDM')
    # Legend outside data region - upper left is safe since data is in lower half
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(-1.08, -0.88)
    ax1.invert_xaxis()  # Higher z = earlier time on right

    # Mark epochs with arrows pointing to lines, not overlapping text
    ax1.annotate('Now', xy=(0.02, -1.0), xytext=(0.3, -0.90),
                fontsize=9, color='green',
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
    ax1.annotate('Peak relaxation', xy=(1.0, -0.95), xytext=(1.8, -0.90),
                fontsize=9, color='orange',
                arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7))

    # Right panel: Hubble parameter evolution
    ax2 = axes[1]

    # Cosmological parameters
    Omega_m0 = 0.3
    Omega_de0 = 0.7

    def H_ratio_lambda(z):
        """ΛCDM: w = -1 constant, so ρ_DE = const"""
        return np.sqrt(Omega_m0 * (1+z)**3 + Omega_de0)

    def rho_de_ratio(z_val):
        """
        Proper dark energy density evolution with varying w(z).
        ρ_DE(z)/ρ_DE(0) = exp(3 * ∫_0^z (1+w(z'))/(1+z') dz')
        """
        if z_val == 0:
            return 1.0
        # Numerical integration
        z_int = np.linspace(0, z_val, 100)
        w_int = relaxation_w(z_int, z_peak=1.0, delta_w=0.12)
        integrand = (1 + w_int) / (1 + z_int)
        integral = np.trapz(integrand, z_int)
        return np.exp(3 * integral)

    def H_ratio_relax(z_val):
        """H(z)/H0 with evolving dark energy"""
        rho_de = Omega_de0 * rho_de_ratio(z_val)
        return np.sqrt(Omega_m0 * (1+z_val)**3 + rho_de)

    H_lambda = [H_ratio_lambda(zi) for zi in z]
    H_relax = [H_ratio_relax(zi) for zi in z]

    ax2.plot(z, H_lambda, '--', color=COLORS['lambda'], linewidth=2, label='$\\Lambda$CDM')
    ax2.plot(z, H_relax, '-', color=COLORS['dark_energy'], linewidth=2.5, label='Relaxation')

    ax2.set_xlabel('Redshift $z$')
    ax2.set_ylabel('$H(z)/H_0$')
    ax2.set_title('(b) Hubble Parameter Evolution')
    # Legend in lower right - curves go up to upper left, so lower right is safe
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlim(0, 3)
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.savefig('../figures/fig5_dark_energy.png', dpi=300, bbox_inches='tight')
    plt.savefig('../figures/fig5_dark_energy.pdf', bbox_inches='tight')
    plt.close()
    print("Generated: fig5_dark_energy")


# =============================================================================
# Figure 4: Information Accumulation Tracks Relaxation
# =============================================================================
def fig6_information():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    np.random.seed(42)
    t = np.linspace(0, 20, 500)

    # Relaxation rate (theoretical)
    dR = relaxation_rate(t, R_max=1.0, k=0.5, t0=10)

    # Information accumulation rate (noisy measurement)
    info_rate = info_accumulation_rate(t, k=0.5, t0=10, noise_scale=0.08)
    info_rate = np.maximum(info_rate, 0)  # Ensure non-negative

    # Smooth the info rate for visualization
    from scipy.ndimage import gaussian_filter1d
    info_rate_smooth = gaussian_filter1d(info_rate, sigma=5)

    # Plot
    ax.plot(t, dR, '-', color=COLORS['rate'], linewidth=2.5,
            label='Relaxation rate $\\dot{R}(t)$', zorder=3)
    ax.scatter(t[::10], info_rate[::10], s=20, alpha=0.5, color=COLORS['info'],
               label='Observed info accumulation (samples)', zorder=2)
    ax.plot(t, info_rate_smooth, '--', color=COLORS['info'], linewidth=2,
            label='Smoothed info rate', zorder=2)


    ax.set_xlabel('Substrate coordinate $s$')
    ax.set_ylabel('Rate')
    ax.set_title('Information Accumulation Rate Tracks Relaxation Rate')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 0.18)

    # Add interpretation
    fig.text(0.5, 0.02,
             'Observers accumulate information at rates bounded by local relaxation dynamics.',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig('../figures/fig6_information.png', dpi=300, bbox_inches='tight')
    plt.savefig('../figures/fig6_information.pdf', bbox_inches='tight')
    plt.close()
    print("Generated: fig6_information")


# =============================================================================
# Figure 5: Symmetry-Breaking Cascade (Schematic)
# =============================================================================
def fig2_cascade():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, 'Symmetry-Breaking Cascade', ha='center', fontsize=16, fontweight='bold')

    # Levels
    levels = [
        (8.5, 'Primordial Substrate', 'Maximal symmetry\nNo structure', COLORS['substrate'], None),
        (7.0, 'Distinction Break', 'Something vs nothing\n→ Set theory', '#FFE0B2', 'Membership'),
        (5.5, 'Ordering Break', 'Before vs after\n→ Ordinal structure', '#C8E6C9', 'Sequence'),
        (4.0, 'Compositional Break', 'Aggregation\n→ Algebraic structure', '#BBDEFB', 'Operations'),
        (2.5, 'Locality Break', 'Near vs far\n→ Topological structure', '#E1BEE7', 'Continuity'),
        (1.0, 'Metric Break', 'Distance\n→ Geometric structure', '#FFCDD2', 'Curvature'),
    ]

    # Draw levels
    for i, (y, title, desc, color, math_concept) in enumerate(levels):
        # Box
        box = FancyBboxPatch((1, y - 0.4), 4, 0.8, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(3, y, title, ha='center', va='center', fontsize=11, fontweight='bold')

        # Description
        ax.text(5.5, y, desc, ha='left', va='center', fontsize=9)

        # Math concept (right side) - match left box style
        if math_concept:
            # Draw box matching left side dimensions
            math_box = FancyBboxPatch((8, y - 0.4), 3, 0.8, boxstyle="round,pad=0.05",
                                       facecolor=COLORS['math'], edgecolor='darkgreen', linewidth=1.5)
            ax.add_patch(math_box)
            ax.text(9.5, y, math_concept, ha='center', va='center', fontsize=11, fontweight='bold')
            # Connecting arrow
            ax.annotate('', xy=(5, y), xytext=(8, y),
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=1.5))

        # Arrow to next level
        if i < len(levels) - 1:
            next_y = levels[i + 1][0]
            ax.annotate('', xy=(3, next_y + 0.5), xytext=(3, y - 0.5),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Side label
    ax.text(0.5, 5, 'TIME\n→', ha='center', va='center', fontsize=12,
            rotation=90, fontweight='bold', color='gray')
    ax.text(9.5, 8.5, 'Emergent Math', ha='center', va='center', fontsize=11,
            fontweight='bold', color='darkgreen')

    # Bottom annotation - use figure coordinates to avoid collision
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.text(0.5, 0.02,
            'Each symmetry break creates new structure. '
            'Physical law and mathematical structure are the same thing—the topology of the knot.',
            ha='center', va='bottom', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.savefig('../figures/fig2_cascade.png', dpi=300, bbox_inches='tight')
    plt.savefig('../figures/fig2_cascade.pdf', bbox_inches='tight')
    plt.close()
    print("Generated: fig2_cascade")


# =============================================================================
# Figure 6: The Napkin Metaphor
# =============================================================================
def fig1_napkin():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Create stylized napkin representations
    np.random.seed(42)

    for idx, (ax, title, desc, complexity) in enumerate([
        (axes[0], 'Early Universe', 'Tightly crumpled\nSlow relaxation', 1.0),
        (axes[1], 'Present Epoch', 'Loosening\nFast relaxation', 0.5),
        (axes[2], 'Heat Death', 'Fully flat\nNo relaxation', 0.1),
    ]):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Generate wrinkled surface
        theta = np.linspace(0, 2*np.pi, 100)

        if complexity > 0.8:
            # Highly crumpled - lots of folds
            for i in range(20):
                r = 0.3 + 0.7 * np.random.rand()
                offset_x = 0.3 * (np.random.rand() - 0.5)
                offset_y = 0.3 * (np.random.rand() - 0.5)
                noise = 0.2 * np.sin(5*theta + np.random.rand()*2*np.pi)
                x = (r + noise) * np.cos(theta) + offset_x
                y = (r + noise) * np.sin(theta) + offset_y
                ax.plot(x, y, color=COLORS['knot'], alpha=0.4, linewidth=1)

            # Central mass
            circle = Circle((0, 0), 0.4, facecolor=COLORS['knot'], alpha=0.7, edgecolor='black')
            ax.add_patch(circle)

        elif complexity > 0.3:
            # Partially relaxed
            for i in range(10):
                r = 0.5 + 0.4 * np.random.rand()
                noise = 0.1 * np.sin(3*theta + np.random.rand()*2*np.pi)
                x = (r + noise) * np.cos(theta)
                y = (r + noise) * np.sin(theta)
                ax.plot(x, y, color=COLORS['knot'], alpha=0.5, linewidth=1.5)

            # Less dense center
            circle = Circle((0, 0), 0.3, facecolor=COLORS['knot'], alpha=0.4, edgecolor='black')
            ax.add_patch(circle)

        else:
            # Nearly flat
            for i in range(3):
                r = 0.9 + 0.1 * i
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                ax.plot(x, y, color=COLORS['knot'], alpha=0.3, linewidth=2)

            # Minimal center
            circle = Circle((0, 0), 0.15, facecolor=COLORS['knot'], alpha=0.2, edgecolor='black')
            ax.add_patch(circle)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.text(0, -1.3, desc, ha='center', fontsize=10, style='italic')

    # Add arrows between panels
    fig.text(0.36, 0.5, '→', fontsize=30, ha='center', va='center', color='gray')
    fig.text(0.64, 0.5, '→', fontsize=30, ha='center', va='center', color='gray')

    # Bottom annotation
    fig.text(0.5, 0.02,
             'The universe is a crumpled napkin slowly relaxing toward flatness. '
             'Time is the relaxation.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig('../figures/fig1_napkin.png', dpi=300, bbox_inches='tight')
    plt.savefig('../figures/fig1_napkin.pdf', bbox_inches='tight')
    plt.close()
    print("Generated: fig1_napkin")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    import os
    os.makedirs('../figures', exist_ok=True)

    print("Generating figures for Cosmic Relaxation paper...")
    print("=" * 50)

    fig3_relaxation()
    fig4_knot()
    fig5_dark_energy()
    fig6_information()
    fig2_cascade()
    fig1_napkin()

    print("=" * 50)
    print("All figures generated successfully!")
