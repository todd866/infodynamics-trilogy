"""
Black Hole Aperture Simulation

Demonstrates that observer-relative dimensional apertures produce
horizon-like phenomenology (time dilation, complementarity) without
invoking GR explicitly.

Key insight: The same high-dimensional dynamics appear differently
to observers with different apertures. External observers see time
freeze at the horizon; infalling observers see nothing special.

Fixed issues (Dec 2025):
- τ̇ now computed WITHOUT normalization, so it actually → 0 as aperture closes
- k_dyn computed from covariance eigenvalues (not just weights)
- S_acc computed from actual covariance over sliding window
- Radius range extended to r_min=0.001 so k_w actually approaches ~2
- Figure 4 compares τ̇ to Schwarzschild, not k_eff
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import welch
from dataclasses import dataclass
from typing import Tuple, List
from collections import deque
import os

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)


@dataclass
class ObserverState:
    """State variables for an observer."""
    k_w: float           # Channel participation (from weights)
    k_dyn: float         # Dynamical dimension (from covariance eigenvalues)
    s_acc: float         # Accessible entropy (log det C)
    tau_rate: float      # Correlation accumulation rate (geodesic speed)
    q_cumulative: float  # Thermodynamic cost of erasure
    tau_accumulated: float  # Total accumulated proper time


class CoupledOscillatorSystem:
    """
    High-dimensional dynamical system: N coupled damped oscillators.

    This serves as the "underlying reality" that both observers watch.
    The dynamics are the same; only the apertures differ.

    Note: Includes damping (coupling to thermal bath), so not purely Hamiltonian.
    """

    def __init__(self, n_oscillators: int = 50, coupling: float = 0.1,
                 damping: float = 0.01):
        self.n = n_oscillators
        self.coupling = coupling
        self.damping = damping
        self.reset()

    def reset(self):
        """Reset to random initial conditions (N(0,1) as stated in paper)."""
        self.positions = np.random.randn(self.n)
        self.velocities = np.random.randn(self.n)

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        """Compute derivatives for ODE integration."""
        x = state[:self.n]
        v = state[self.n:]

        # Spring forces (harmonic)
        forces = -x

        # Coupling to neighbors
        forces[:-1] += self.coupling * (x[1:] - x[:-1])
        forces[1:] += self.coupling * (x[:-1] - x[1:])

        # Damping (coupling to bath)
        forces -= self.damping * v

        return np.concatenate([v, forces])

    def step(self, dt: float = 0.01):
        """Advance system by one timestep."""
        state = np.concatenate([self.positions, self.velocities])
        t_span = [0, dt]
        result = odeint(self.derivatives, state, t_span)
        self.positions = result[-1, :self.n]
        self.velocities = result[-1, self.n:]

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current positions and velocities."""
        return self.positions.copy(), self.velocities.copy()


class Observer:
    """
    An observer with a specific aperture (access to degrees of freedom).

    The aperture determines which modes the observer can see.
    External observers lose access to high-frequency modes near the horizon.
    Infalling observers maintain full access.
    """

    def __init__(self, n_modes: int, observer_type: str = 'infalling',
                 window_size: int = 200):  # Increased for proper covariance estimation
        self.n = n_modes
        self.observer_type = observer_type
        self.window_size = window_size
        self.warmup_samples = 100  # Don't report until warmed up

        # Sliding window for covariance estimation (positions only, not 2N)
        self.position_history = deque(maxlen=window_size)

        self.prev_s_acc = None
        self.q_cumulative = 0.0
        self.tau_accumulated = 0.0
        self.baseline_s_acc = None
        self.samples_seen = 0

    def get_aperture_weights(self, radius: float) -> np.ndarray:
        """
        Compute aperture weights based on radius.

        For external observer: weights decrease for high-frequency modes
        as radius approaches 0 (horizon).

        For infalling observer: full access always.
        """
        if self.observer_type == 'infalling':
            return np.ones(self.n)

        # External observer: high-frequency modes suppressed near horizon
        # Use steeper suppression to actually reach k_w ~ 2
        mode_freqs = np.arange(1, self.n + 1) / self.n
        weights = np.power(radius, mode_freqs * 5)  # Steeper exponent
        return weights

    def compute_k_w(self, weights: np.ndarray) -> float:
        """Compute channel participation (from weights alone)."""
        total = np.sum(weights)
        total_sq = np.sum(weights ** 2)
        if total_sq < 1e-10:
            return 1.0
        return (total ** 2) / total_sq

    def compute_k_dyn(self, positions: np.ndarray, velocities: np.ndarray,
                      weights: np.ndarray) -> float:
        """
        Compute dynamical dimension from observed covariance eigenvalues.

        Uses POSITIONS ONLY (N dimensions, not 2N) to ensure well-posed
        covariance estimation with the sliding window.

        Uses SVD-based effective rank: only counts eigenvalues above a
        relative threshold, avoiding artifacts from regularization.
        """
        # Observed positions (weighted) - NOT velocities, to keep dimension N
        x_obs = weights * positions
        self.position_history.append(x_obs)
        self.samples_seen += 1

        # Need enough samples for meaningful covariance
        if self.samples_seen < self.warmup_samples:
            return self.compute_k_w(weights)  # Fallback during warmup

        # Compute covariance from history
        states = np.array(self.position_history)

        # Use SVD for numerical stability
        # Center the data
        states_centered = states - states.mean(axis=0)

        # SVD of data matrix
        try:
            U, S, Vt = np.linalg.svd(states_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return self.compute_k_w(weights)

        # Eigenvalues of covariance are S^2 / (n-1)
        eigenvalues = (S ** 2) / (len(states) - 1)

        # Only use eigenvalues above relative threshold (effective rank)
        threshold = 1e-6 * eigenvalues.max() if eigenvalues.max() > 0 else 1e-10
        significant_eigs = eigenvalues[eigenvalues > threshold]

        if len(significant_eigs) < 2:
            return 1.0

        # Participation ratio on significant eigenvalues only
        total = np.sum(significant_eigs)
        total_sq = np.sum(significant_eigs ** 2)
        return (total ** 2) / total_sq

    def compute_s_acc(self, positions: np.ndarray, velocities: np.ndarray,
                      weights: np.ndarray) -> float:
        """
        Compute accessible entropy: S_acc = (1/2) log det C_obs

        Uses POSITIONS ONLY from sliding window (same as k_dyn).
        Uses SVD-based log-det on the effective subspace only.
        Returns value relative to baseline after warmup.
        """
        if self.samples_seen < self.warmup_samples:
            # During warmup, return 0 (will be ignored in Q calculation)
            return 0.0

        # Use position history (already populated by compute_k_dyn)
        states = np.array(self.position_history)
        states_centered = states - states.mean(axis=0)

        # SVD for log-det
        try:
            U, S, Vt = np.linalg.svd(states_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0

        # Eigenvalues of covariance
        eigenvalues = (S ** 2) / (len(states) - 1)

        # Only use significant eigenvalues
        threshold = 1e-6 * eigenvalues.max() if eigenvalues.max() > 0 else 1e-10
        significant_eigs = eigenvalues[eigenvalues > threshold]

        if len(significant_eigs) < 1:
            return 0.0

        # log det = sum of log eigenvalues (in effective subspace)
        s_acc = 0.5 * np.sum(np.log(significant_eigs))

        # Set baseline on first valid computation
        if self.baseline_s_acc is None:
            self.baseline_s_acc = s_acc

        # Return relative to baseline (so we see aperture-driven changes)
        return s_acc - self.baseline_s_acc

    def compute_tau_rate(self, velocities: np.ndarray,
                         weights: np.ndarray) -> float:
        """
        Compute correlation accumulation rate (geodesic speed).

        τ̇ = sqrt(v^T G v) where G = diag(w) is the induced metric.

        NO NORMALIZATION: when w → 0, τ̇ → 0 (time freezes).
        """
        # This is the key fix: no division by sum(weights)
        weighted_v_sq = np.sum(weights * velocities ** 2)
        return np.sqrt(weighted_v_sq)

    def observe(self, positions: np.ndarray, velocities: np.ndarray,
                radius: float, dt: float = 1.0) -> ObserverState:
        """
        Make an observation and compute all state variables.
        """
        weights = self.get_aperture_weights(radius)

        k_w = self.compute_k_w(weights)
        k_dyn = self.compute_k_dyn(positions, velocities, weights)
        s_acc = self.compute_s_acc(positions, velocities, weights)
        tau_rate = self.compute_tau_rate(velocities, weights)

        # Thermodynamic cost: erasure when S_acc drops
        # Only count after warmup (when s_acc is meaningful)
        if self.samples_seen > self.warmup_samples and self.prev_s_acc is not None:
            delta_s = max(0, self.prev_s_acc - s_acc)
            self.q_cumulative += delta_s
        if self.samples_seen > self.warmup_samples:
            self.prev_s_acc = s_acc

        # Accumulate proper time
        self.tau_accumulated += tau_rate * dt

        return ObserverState(
            k_w=k_w,
            k_dyn=k_dyn,
            s_acc=s_acc,
            tau_rate=tau_rate,
            q_cumulative=self.q_cumulative,
            tau_accumulated=self.tau_accumulated
        )


def run_simulation(n_steps: int = 1000, n_oscillators: int = 50,
                   radius_profile: str = 'static',
                   r_min: float = 0.001) -> dict:
    """
    Run the full simulation with both observers.

    Args:
        n_steps: Number of timesteps
        n_oscillators: Number of coupled oscillators
        radius_profile: 'static', 'infall', or 'merger'
        r_min: Minimum radius (set very small to reach k_w ~ 2)

    Returns:
        Dictionary with all time series data
    """
    system = CoupledOscillatorSystem(n_oscillators)
    external = Observer(n_oscillators, 'external')
    infalling = Observer(n_oscillators, 'infalling')

    # Storage
    data = {
        'time': [],
        'radius': [],
        'external': {'k_w': [], 'k_dyn': [], 's_acc': [], 'tau_rate': [],
                     'q': [], 'tau': []},
        'infalling': {'k_w': [], 'k_dyn': [], 's_acc': [], 'tau_rate': [],
                      'q': [], 'tau': []},
    }

    for t in range(n_steps):
        # Determine radius based on profile
        if radius_profile == 'static':
            radius = 0.5
        elif radius_profile == 'infall':
            # Gradual approach to horizon - go much closer
            radius = max(r_min, 1.0 - t / n_steps * (1.0 - r_min))
        elif radius_profile == 'merger':
            if t < n_steps // 3:
                radius = 1.0 - 0.5 * (t / (n_steps // 3))
            elif t < 2 * n_steps // 3:
                phase = (t - n_steps // 3) / (n_steps // 3)
                radius = 0.5 - 0.45 * phase
            else:
                phase = (t - 2 * n_steps // 3) / (n_steps // 3)
                radius = 0.05 + 0.1 * np.exp(-3 * phase) * np.sin(20 * phase)
                radius = max(r_min, radius)
        else:
            radius = 0.5

        # Step the system
        system.step()
        pos, vel = system.get_state()

        # Both observers watch
        ext_state = external.observe(pos, vel, radius)
        inf_state = infalling.observe(pos, vel, 1.0)

        # Store
        data['time'].append(t)
        data['radius'].append(radius)

        for key in ['k_w', 'k_dyn', 's_acc', 'tau_rate', 'q', 'tau']:
            attr = key if key not in ['q', 'tau'] else ('q_cumulative' if key == 'q' else 'tau_accumulated')
            data['external'][key].append(getattr(ext_state, attr))
            data['infalling'][key].append(getattr(inf_state, attr))

    # Convert to numpy
    for key in data:
        if isinstance(data[key], list):
            data[key] = np.array(data[key])
        elif isinstance(data[key], dict):
            for subkey in data[key]:
                data[key][subkey] = np.array(data[key][subkey])

    return data


def generate_figure_1(data: dict, filename: str = '../figures/fig1_time_dilation.pdf'):
    """
    Figure 1: Time dilation demonstration.

    Shows k_w, k_dyn, τ̇, and accumulated τ for both observers during infall.
    External observer's clock freezes; infalling continues.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: Radius profile
    ax = axes[0, 0]
    ax.plot(data['time'], data['radius'], 'k-', linewidth=2)
    ax.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='Near-horizon')
    ax.set_xlabel('Coordinate time')
    ax.set_ylabel('Radius')
    ax.set_title('A. Infall trajectory')
    ax.set_yscale('log')
    ax.legend()

    # Panel B: Both k_w and k_dyn for BOTH observers
    ax = axes[0, 1]
    ax.plot(data['time'], data['external']['k_w'], 'r-',
            label='External $k_w$', linewidth=2)
    ax.plot(data['time'], data['external']['k_dyn'], 'r--',
            label='External $k_{dyn}$', linewidth=1.5, alpha=0.7)
    ax.plot(data['time'], data['infalling']['k_w'], 'c-',
            label='Infalling $k_w$', linewidth=2)
    ax.plot(data['time'], data['infalling']['k_dyn'], 'c--',
            label='Infalling $k_{dyn}$', linewidth=1.5, alpha=0.7)
    ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5)
    ax.text(data['time'][-1]*0.05, 4, 'Low-$k$ regime', fontsize=8, color='gray')
    ax.set_xlabel('Coordinate time')
    ax.set_ylabel('Effective dimension')
    ax.set_title('B. Dimensional collapse ($k_w$ and $k_{dyn}$)')
    ax.legend(fontsize=8, loc='upper right')

    # Panel C: Correlation rate (τ̇)
    ax = axes[1, 0]
    ax.plot(data['time'], data['external']['tau_rate'], 'r-',
            label='External', linewidth=2)
    ax.plot(data['time'], data['infalling']['tau_rate'], 'c-',
            label='Infalling', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Coordinate time')
    ax.set_ylabel('$\\dot{\\tau}$ (correlation rate)')
    ax.set_title('C. Time flow rate (→ 0 at horizon)')
    ax.legend()

    # Panel D: Accumulated proper time
    ax = axes[1, 1]
    ax.plot(data['time'], data['external']['tau'], 'r-',
            label='External', linewidth=2)
    ax.plot(data['time'], data['infalling']['tau'], 'c-',
            label='Infalling', linewidth=2)
    ax.set_xlabel('Coordinate time')
    ax.set_ylabel('$\\tau$ (accumulated proper time)')
    ax.set_title('D. Proper time (external asymptotes)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def generate_figure_2(data: dict, filename: str = '../figures/fig2_thermodynamics.pdf'):
    """
    Figure 2: Thermodynamic cost of aperture squeezing.

    Shows accessible entropy and Landauer erasure proxy.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Accessible entropy (relative to baseline)
    ax = axes[0]
    ax.plot(data['time'], data['external']['s_acc'], 'r-',
            label='External', linewidth=2)
    ax.plot(data['time'], data['infalling']['s_acc'], 'c-',
            label='Infalling', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Coordinate time')
    ax.set_ylabel('$\\Delta S_{acc}$ (relative to baseline)')
    ax.set_title('A. Accessible entropy change')
    ax.legend()

    # Panel B: Cumulative erasure cost
    ax = axes[1]
    ax.plot(data['time'], data['external']['q'], 'r-',
            label='External', linewidth=2)
    ax.plot(data['time'], data['infalling']['q'], 'c-',
            label='Infalling', linewidth=2)
    ax.set_xlabel('Coordinate time')
    ax.set_ylabel('$Q$ (Landauer proxy, nats)')
    ax.set_title('B. Cumulative erasure cost')
    ax.legend()

    # Panel C: τ vs Q (time-cost tradeoff)
    ax = axes[2]
    ax.plot(data['external']['q'], data['external']['tau'], 'r-',
            label='External', linewidth=2)
    ax.plot(data['infalling']['q'], data['infalling']['tau'], 'c-',
            label='Infalling', linewidth=2)
    ax.set_xlabel('Cumulative erasure $Q$')
    ax.set_ylabel('Accumulated proper time $\\tau$')
    ax.set_title('C. Time-cost tradeoff')
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def generate_figure_3(filename: str = '../figures/fig3_ligo_connection.pdf'):
    """
    Figure 3: Connection to gravitational waves (qualitative).

    Shows how aperture dynamics during merger produce
    characteristic ringdown-like waveforms.
    """
    data = run_simulation(n_steps=1500, radius_profile='merger')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: Merger trajectory
    ax = axes[0, 0]
    ax.plot(data['time'], data['radius'], 'k-', linewidth=2)
    ax.axvline(x=500, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1000, color='gray', linestyle='--', alpha=0.5)
    ax.text(250, 0.9, 'Inspiral', ha='center')
    ax.text(750, 0.9, 'Merger', ha='center')
    ax.text(1250, 0.9, 'Ringdown', ha='center')
    ax.set_xlabel('Coordinate time')
    ax.set_ylabel('Effective radius')
    ax.set_title('A. Merger trajectory')

    # Panel B: k_dyn during merger
    ax = axes[0, 1]
    ax.plot(data['time'], data['external']['k_dyn'], 'r-', linewidth=1.5)
    ax.axvline(x=500, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1000, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Coordinate time')
    ax.set_ylabel('$k_{dyn}$ (external)')
    ax.set_title('B. Dimensional collapse during merger')

    # Panel C: τ̇ derivative as signal proxy
    ax = axes[1, 0]
    tau_rate = data['external']['tau_rate']
    signal_proxy = np.gradient(tau_rate)
    ax.plot(data['time'], signal_proxy, 'purple', linewidth=1)
    ax.axvline(x=500, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1000, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Coordinate time')
    ax.set_ylabel('$d\\dot{\\tau}/dt$ (signal proxy)')
    ax.set_title('C. Aperture perturbation signal')

    # Panel D: Power spectrum of ringdown
    ax = axes[1, 1]
    ringdown_start = 1000
    ringdown_signal = signal_proxy[ringdown_start:]
    if len(ringdown_signal) > 10:
        freqs, psd = welch(ringdown_signal, fs=1.0,
                           nperseg=min(256, len(ringdown_signal)//2))
        ax.semilogy(freqs, psd, 'purple', linewidth=2)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power spectral density')
    ax.set_title('D. Ringdown spectrum')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def generate_figure_4(filename: str = '../figures/fig4_k_vs_radius.pdf'):
    """
    Figure 4: τ̇ vs radius compared to Schwarzschild.

    Now compares the actual time dilation proxy (τ̇/τ̇_max) to sqrt(1-r_s/r),
    NOT k_eff/k_max.
    """
    radii = np.linspace(0.001, 1.0, 200)
    n_oscillators = 50

    tau_rates = []
    k_ws = []
    for r in radii:
        mode_freqs = np.arange(1, n_oscillators + 1) / n_oscillators
        weights = np.power(r, mode_freqs * 5)  # Match simulation

        # k_w for reference
        total = np.sum(weights)
        total_sq = np.sum(weights ** 2)
        k_w = (total ** 2) / total_sq
        k_ws.append(k_w)

        # τ̇ proxy: sqrt(sum w_i * v_i^2) with unit velocities
        # Since v_i are dynamic, use expected value: τ̇ ∝ sqrt(sum w_i)
        tau_rate_proxy = np.sqrt(np.sum(weights))
        tau_rates.append(tau_rate_proxy)

    tau_rates = np.array(tau_rates)
    k_ws = np.array(k_ws)

    # Normalize
    tau_normalized = tau_rates / tau_rates[-1]  # Normalize to r=1

    # Schwarzschild: sqrt(1 - r_s/r), with horizon at r_s
    # Choose r_s so behavior matches qualitatively
    r_s = 0.001  # Match our r_min
    schwarzschild = np.sqrt(np.maximum(0, 1 - r_s / radii))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: τ̇ comparison
    ax = axes[0]
    ax.plot(radii, tau_normalized, 'r-', linewidth=2,
            label='$\\dot{\\tau}/\\dot{\\tau}_{\\infty}$ (aperture model)')
    ax.plot(radii, schwarzschild, 'k--', linewidth=2,
            label='$\\sqrt{1 - r_s/r}$ (Schwarzschild)')
    ax.axvline(x=r_s, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Radius $r$')
    ax.set_ylabel('Time dilation factor')
    ax.set_title('A. Time dilation: $\\dot{\\tau}$ vs Schwarzschild')
    ax.legend()
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 1.1)

    # Right: k_w collapse
    ax = axes[1]
    ax.plot(radii, k_ws, 'r-', linewidth=2)
    ax.axhline(y=2, color='gray', linestyle=':', alpha=0.7)
    ax.text(0.5, 2.5, '$k_w = 2$ (surface-confined)', fontsize=9)
    ax.set_xlabel('Radius $r$')
    ax.set_ylabel('Channel participation $k_w$')
    ax.set_title('B. Dimensional collapse')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_oscillators + 5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


if __name__ == '__main__':
    print("Running black hole aperture simulations...")
    print("(Fixed version: τ̇ → 0, k_dyn from covariance, proper S_acc)")

    # Generate main infall simulation
    print("\n1. Running infall simulation...")
    data_infall = run_simulation(n_steps=1000, radius_profile='infall', r_min=0.001)

    print("\n2. Generating Figure 1: Time dilation...")
    generate_figure_1(data_infall)

    print("\n3. Generating Figure 2: Thermodynamics...")
    generate_figure_2(data_infall)

    print("\n4. Generating Figure 3: LIGO connection...")
    generate_figure_3()

    print("\n5. Generating Figure 4: τ̇ vs radius...")
    generate_figure_4()

    print("\nAll figures generated successfully!")
    print("\nKey results:")
    print(f"  - Final k_w (external): {data_infall['external']['k_w'][-1]:.2f}")
    print(f"  - Final τ̇ (external): {data_infall['external']['tau_rate'][-1]:.4f}")
    print(f"  - External τ accumulated: {data_infall['external']['tau'][-1]:.1f}")
    print(f"  - Infalling τ accumulated: {data_infall['infalling']['tau'][-1]:.1f}")
    if data_infall['external']['tau'][-1] > 0.1:
        ratio = data_infall['infalling']['tau'][-1] / data_infall['external']['tau'][-1]
        print(f"  - Time dilation ratio: {ratio:.1f}x")
    else:
        print(f"  - External time essentially frozen (τ̇ → 0)")
