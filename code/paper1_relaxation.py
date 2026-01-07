from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def figs_dir(self) -> Path:
        return self.root / "figures"


def ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate figures. "
            "Install it in your environment and re-run."
        ) from exc


def save_pdf(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def fig1_geometric_maintenance(paths: Paths) -> None:
    """Schematic of the geometric maintenance bound."""
    ensure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.0, 4.0))
    ax.set_axis_off()

    # Colors
    c_substrate = "#e0f2fe"  # light blue
    c_projection = "#fef3c7"  # light yellow
    c_bath = "#fee2e2"  # light red

    box_style = dict(linewidth=2, edgecolor="#374151")

    # Boxes
    substrate = FancyBboxPatch(
        (0.02, 0.25), 0.28, 0.60,
        boxstyle="round,pad=0.02",
        facecolor=c_substrate,
        **box_style
    )
    projection = FancyBboxPatch(
        (0.38, 0.25), 0.26, 0.60,
        boxstyle="round,pad=0.02",
        facecolor=c_projection,
        **box_style
    )
    bath = FancyBboxPatch(
        (0.72, 0.25), 0.26, 0.60,
        boxstyle="round,pad=0.02",
        facecolor=c_bath,
        **box_style
    )

    for patch in (substrate, projection, bath):
        ax.add_patch(patch)

    # Labels
    ax.text(0.16, 0.70, "High-D Substrate", ha="center", va="center",
            fontsize=12, fontweight="bold")
    ax.text(0.16, 0.55, r"$X \in \mathbb{R}^D$", ha="center", va="center",
            fontsize=11)
    ax.text(0.16, 0.40, "asymmetric\n(low symmetry)", ha="center", va="center",
            fontsize=9, color="#555", style="italic")

    ax.text(0.51, 0.70, "Low-D Representation", ha="center", va="center",
            fontsize=12, fontweight="bold")
    ax.text(0.51, 0.55, r"$Y = \Phi(X) \in \mathbb{R}^k$", ha="center", va="center",
            fontsize=11)
    ax.text(0.51, 0.40, "dimensional\nreduction", ha="center", va="center",
            fontsize=9, color="#555", style="italic")

    ax.text(0.85, 0.70, "Heat Bath", ha="center", va="center",
            fontsize=12, fontweight="bold")
    ax.text(0.85, 0.55, r"$T$", ha="center", va="center", fontsize=14)
    ax.text(0.85, 0.40, "entropy\nincrease", ha="center", va="center",
            fontsize=9, color="#555", style="italic")

    # Arrow substrate -> projection
    ax.add_patch(
        FancyArrowPatch(
            (0.30, 0.55),
            (0.38, 0.55),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2,
            color="#1e40af",
        )
    )
    ax.text(0.34, 0.68, r"$\Phi$", ha="center", va="center", fontsize=12,
            fontweight="bold", color="#1e40af")

    # Arrow projection -> bath
    ax.add_patch(
        FancyArrowPatch(
            (0.64, 0.55),
            (0.72, 0.55),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2,
            color="#b91c1c",
        )
    )
    ax.text(0.68, 0.68, r"$Q$", ha="center", va="center", fontsize=12,
            fontweight="bold", color="#b91c1c")

    # Bound equation
    bound = r"$W_{\mathrm{diss,min}} \geq k_B T\,(\ln 2\,\Delta I + C_\Phi)$"
    ax.text(0.50, 0.12, bound, ha="center", va="center", fontsize=15,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                     edgecolor="#374151", linewidth=1.5))

    # Term labels
    ax.text(
        0.32, 0.02,
        r"$\Delta I$: information removed",
        ha="center", va="center", fontsize=10, color="#1e40af",
    )
    ax.text(
        0.68, 0.02,
        r"$C_\Phi$: geometric contraction cost",
        ha="center", va="center", fontsize=10, color="#b91c1c",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 0.90)

    save_pdf(fig, paths.figs_dir / "fig1_geometric_maintenance.pdf")
    plt.close(fig)


def fig2_dimensional_relaxation(paths: Paths) -> None:
    """
    Thermal relaxation: asymmetric → symmetric.

    Shows D_eff increasing and I_struct decreasing as system relaxes to isotropy.
    """
    ensure_matplotlib()
    import matplotlib.pyplot as plt

    D = 20
    sigma_par2_0 = 1.0
    sigma_perp2_0 = 1e-4
    diffusion = 1.0

    t = np.logspace(-3, 2, 400)
    lam1 = sigma_par2_0 + 2.0 * diffusion * t
    lam = sigma_perp2_0 + 2.0 * diffusion * t

    # Participation ratio / effective dimension.
    tr = lam1 + (D - 1) * lam
    tr2 = lam1**2 + (D - 1) * lam**2
    d_eff = (tr**2) / tr2

    # Structure-information: KL(p || p_iso) where p_iso has Sigma_iso = (tr/D) I.
    lam_iso = tr / D
    logdet = np.log(lam1) + (D - 1) * np.log(lam)
    logdet_iso = D * np.log(lam_iso)
    istruct_nats = 0.5 * (logdet_iso - logdet)
    istruct_bits = istruct_nats / np.log(2.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.5))

    # Left panel: D_eff
    color_dim = "#2563eb"
    ax1.plot(t, d_eff, color=color_dim, linewidth=2.5)
    ax1.axhline(D, color="#94a3b8", linestyle="--", linewidth=1, label=f"$D = {D}$ (isotropic)")
    ax1.axhline(1, color="#94a3b8", linestyle=":", linewidth=1)
    ax1.set_xscale("log")
    ax1.set_xlim(t.min(), t.max())
    ax1.set_ylim(0.5, D + 1)
    ax1.set_xlabel(r"time $t$ (normalized, $D_{\mathrm{diff}}=1$)", fontsize=11)
    ax1.set_ylabel(r"effective dimension $D_{\mathrm{eff}}$", fontsize=11, color=color_dim)
    ax1.tick_params(axis="y", labelcolor=color_dim)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_title("Symmetry increases", fontsize=12, fontweight="bold")

    # Annotations
    ax1.annotate("asymmetric\n(low $D_{\\mathrm{eff}}$)",
                xy=(t[20], d_eff[20]), xytext=(0.003, 8),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1))
    ax1.annotate("symmetric\n(high $D_{\\mathrm{eff}}$)",
                xy=(t[-30], d_eff[-30]), xytext=(30, 12),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1))

    # Right panel: I_struct
    color_info = "#dc2626"
    ax2.plot(t, istruct_bits, color=color_info, linewidth=2.5)
    ax2.axhline(0, color="#94a3b8", linestyle="--", linewidth=1)
    ax2.set_xscale("log")
    ax2.set_xlim(t.min(), t.max())
    ax2.set_ylim(-0.5, float(istruct_bits.max()) * 1.08)
    ax2.set_xlabel(r"time $t$ (normalized, $D_{\mathrm{diff}}=1$)", fontsize=11)
    ax2.set_ylabel(r"structure-information $I_{\mathrm{struct}}$ (bits)", fontsize=11, color=color_info)
    ax2.tick_params(axis="y", labelcolor=color_info)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title("Information entropy decreases", fontsize=12, fontweight="bold")

    # Annotations
    ax2.annotate("high $I_{\\mathrm{struct}}$\n(asymmetric)",
                xy=(t[20], istruct_bits[20]), xytext=(0.02, istruct_bits.max()*0.75),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1))
    ax2.annotate("$I_{\\mathrm{struct}} \\to 0$\n(symmetric)",
                xy=(t[-30], istruct_bits[-30]), xytext=(10, 8),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1))

    # Add caption
    caption = "\n".join([
        rf"$D={D}$, $\Sigma_0 = \mathrm{{diag}}({sigma_par2_0:.0f}, {sigma_perp2_0:.0e},\ldots)$",
        r"Diffusion: $\Sigma(t)=\Sigma_0 + 2Dt\,I$",
    ])
    fig.text(
        0.50, 0.02, caption, ha="center", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, linewidth=0.0),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    save_pdf(fig, paths.figs_dir / "fig2_dimensional_relaxation.pdf")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = Paths(root=root)

    fig1_geometric_maintenance(paths)
    fig2_dimensional_relaxation(paths)
    print(f"Wrote figures to: {paths.figs_dir}")


if __name__ == "__main__":
    main()
