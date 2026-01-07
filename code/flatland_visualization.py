"""
Flatland Black Hole Visualization

A 3D sphere passing through Flatland (2D plane) serves as an analogy
for dimensional aperture. The 2D observers see only a circle whose
radius changes as the sphere passes through—their "aperture" to the
full 3D structure varies with position.

Near the "horizon" (where the sphere just touches the plane), the
accessible slice shrinks to a point—time would freeze for observers
trying to track correlations in the vanishing cross-section.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Ensure figures directory exists
import os
os.makedirs('../figures', exist_ok=True)


def create_flatland_figure():
    """
    Create a figure showing the Flatland analogy for dimensional aperture.

    A sphere passing through a 2D plane demonstrates how observers with
    limited dimensional access see only a slice of the full dynamics.
    """
    fig = plt.figure(figsize=(14, 5))

    # Three panels: before horizon, at horizon, through horizon
    positions = [-0.8, 0.0, 0.8]  # Sphere center z-positions
    titles = ['Approaching\n(large aperture)',
              'At horizon\n(aperture closing)',
              'Beyond\n(aperture reopening)']

    for idx, (z_center, title) in enumerate(zip(positions, titles)):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

        # Draw the sphere
        R = 1.0  # Sphere radius
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 30)
        x_sphere = R * np.outer(np.cos(u), np.sin(v))
        y_sphere = R * np.outer(np.sin(u), np.sin(v))
        z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v)) + z_center

        # Color the sphere - red for above plane, blue for below
        ax.plot_surface(x_sphere, y_sphere, z_sphere,
                       color='steelblue', alpha=0.6, linewidth=0)

        # Draw the Flatland plane (z=0)
        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 10),
                            np.linspace(-1.5, 1.5, 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

        # Draw the intersection circle (what Flatlanders see)
        if abs(z_center) < R:
            r_intersection = np.sqrt(R**2 - z_center**2)
            theta = np.linspace(0, 2*np.pi, 100)
            x_circle = r_intersection * np.cos(theta)
            y_circle = r_intersection * np.sin(theta)
            z_circle = np.zeros_like(theta)
            ax.plot(x_circle, y_circle, z_circle, 'k-', linewidth=3,
                   label=f'Visible slice (r={r_intersection:.2f})')
        else:
            r_intersection = 0

        # Formatting
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z (hidden dim)')
        ax.set_title(f'{title}\nSlice radius: {r_intersection:.2f}', fontsize=10)
        ax.view_init(elev=20, azim=45)

        # Hide some axis elements for cleaner look
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig('../figures/fig5_flatland.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig('../figures/fig5_flatland.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved ../figures/fig5_flatland.pdf")
    plt.close()


def create_aperture_vs_position():
    """
    Show how the accessible aperture (intersection area) varies
    as the sphere passes through Flatland.

    This is the analog of time dilation: near the horizon (z=±R),
    the accessible slice vanishes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    R = 1.0
    z = np.linspace(-1.5, 1.5, 200)

    # Intersection radius as function of z
    r_slice = np.where(np.abs(z) < R, np.sqrt(R**2 - z**2), 0)

    # "Aperture" = area of slice / area of full sphere cross-section
    aperture = (r_slice / R) ** 2  # Normalized

    # Panel 1: Slice radius vs position
    ax1 = axes[0]
    ax1.fill_between(z, r_slice, alpha=0.3, color='steelblue')
    ax1.plot(z, r_slice, 'b-', linewidth=2)
    ax1.axvline(x=-R, color='red', linestyle='--', alpha=0.7, label='Horizon')
    ax1.axvline(x=R, color='red', linestyle='--', alpha=0.7)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Position (z)', fontsize=11)
    ax1.set_ylabel('Visible slice radius', fontsize=11)
    ax1.set_title('What Flatlanders can access', fontsize=12)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-0.1, 1.2)
    ax1.legend(loc='upper right')

    # Panel 2: Aperture (accessible fraction) vs position
    ax2 = axes[1]
    ax2.fill_between(z, aperture, alpha=0.3, color='coral')
    ax2.plot(z, aperture, 'r-', linewidth=2, label='Dimensional aperture')
    ax2.axvline(x=-R, color='red', linestyle='--', alpha=0.7, label='Horizon')
    ax2.axvline(x=R, color='red', linestyle='--', alpha=0.7)

    # Compare to Schwarzschild-like profile
    # Map z to r: z=0 is far from horizon, z=±R is at horizon
    # Use r = R + |z| mapping, so r_s = R
    r_mapped = R + np.abs(z)
    schwarzschild = np.sqrt(1 - R/r_mapped)
    schwarzschild = np.where(np.abs(z) > 0, schwarzschild, 0)
    ax2.plot(z, schwarzschild, 'k--', linewidth=1.5, alpha=0.7,
            label=r'$\sqrt{1-r_s/r}$ (Schwarzschild)')

    ax2.set_xlabel('Position (z)', fontsize=11)
    ax2.set_ylabel('Accessible fraction', fontsize=11)
    ax2.set_title('Aperture collapses at horizons', fontsize=12)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-0.1, 1.2)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('../figures/fig6_aperture_profile.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig('../figures/fig6_aperture_profile.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved ../figures/fig6_aperture_profile.pdf")
    plt.close()


def create_time_dilation_flatland():
    """
    Show accumulated "proper time" for a Flatlander observer
    as a function of the sphere's passage.

    Time = integral of aperture, so it grows slowly near horizons.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    R = 1.0
    z = np.linspace(-1.5, 1.5, 500)
    dz = z[1] - z[0]

    # Intersection radius
    r_slice = np.where(np.abs(z) < R, np.sqrt(R**2 - z**2), 0)
    aperture = r_slice / R

    # Proper time = cumulative aperture
    tau = np.cumsum(aperture) * dz

    # Coordinate time (linear)
    t_coord = z - z[0]

    ax.plot(t_coord, tau, 'b-', linewidth=2.5, label='Proper time (aperture-limited)')
    ax.plot(t_coord, t_coord, 'k--', linewidth=1.5, alpha=0.5, label='Coordinate time')

    # Mark horizon crossings
    horizon_idx_1 = np.argmin(np.abs(z - (-R)))
    horizon_idx_2 = np.argmin(np.abs(z - R))
    ax.axvline(x=t_coord[horizon_idx_1], color='red', linestyle=':', alpha=0.7)
    ax.axvline(x=t_coord[horizon_idx_2], color='red', linestyle=':', alpha=0.7)
    ax.text(t_coord[horizon_idx_1] + 0.05, 0.5, 'Horizon 1', rotation=90,
            color='red', fontsize=9, alpha=0.8)
    ax.text(t_coord[horizon_idx_2] + 0.05, 0.5, 'Horizon 2', rotation=90,
            color='red', fontsize=9, alpha=0.8)

    ax.set_xlabel('Coordinate time', fontsize=12)
    ax.set_ylabel('Accumulated proper time', fontsize=12)
    ax.set_title('Time dilation in Flatland: proper time lags at horizons', fontsize=12)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/fig7_flatland_time.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.savefig('../figures/fig7_flatland_time.pdf', bbox_inches='tight',
                facecolor='white')
    print("Saved ../figures/fig7_flatland_time.pdf")
    plt.close()


if __name__ == '__main__':
    print("Generating Flatland visualizations...")
    create_flatland_figure()
    create_aperture_vs_position()
    create_time_dilation_flatland()
    print("\nAll Flatland figures generated!")
