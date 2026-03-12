# Infodynamics Trilogy

Three papers extending Vopson's infodynamics framework, scaling from microscopic to cosmological.

## The Arc

| Paper | Scale | Core Claim |
|-------|-------|------------|
| **1. Thermodynamic Foundation** | Microscopic | Asymmetric states cost work to maintain. Structure-information (departure from equilibrium) decays toward symmetry. This is the mechanism behind the second law of infodynamics. |
| **2. Time as Information Rate** | Mesoscopic | Time dilation = channel contraction. An observer's "dimensional aperture" determines their information accumulation rate. Horizons are where apertures close. Black hole phenomenology emerges from information geometry. |
| **3. The Relaxing Knot** | Cosmological | The universe is a constrained configuration ("knot") relaxing toward equilibrium. Time is relaxation rate. Dark energy is relaxation pressure. Mathematics is knot structure. |

## Key Results

### Paper 1: Thermodynamic Foundation
- **Geometric maintenance bound**: $W_{\text{diss,min}} \geq k_B T(\ln 2 \cdot \Delta I + C_\Phi)$
- Structure costs work; symmetric states are thermodynamically free
- Reconciles second law of infodynamics with second law of thermodynamics

### Paper 2: Black Hole Aperture
- **Time proxy**: $\dot{\tau} = \sqrt{\sum_i w_i \dot{x}_i^2}$ (Fisher-speed through accessible state space)
- Simulated coupled oscillators show horizon-like phenomenology without GR
- Complementarity emerges: same dynamics, different apertures, different clocks

### Paper 3: Cosmic Relaxation
- **Relaxation dynamics**: Logistic slow-fast-slow curve predicts dark energy evolution
- Dissolves "unreasonable effectiveness of mathematics" - math is physics, both are knot structure
- Connects microscopic (Landauer) to cosmological (dark energy) through constraint release

## Simulations

All figures are generated from reproducible Python code:

```bash
pip install -r requirements.txt

python paper1_simulations.py   # Paper 1 figures
python paper2_simulations.py   # Paper 2 figures
python paper3_simulations.py   # Paper 3 figures
```

### Key Results

| Paper | Key Simulation | Result |
|-------|---------------|--------|
| 1 | Landauer illustration | R² = 0.97 bound scaling |
| 2 | Schwarzschild comparison | R² = 0.97 match to GR |
| 2 | Complementarity | 3.4× time dilation across observers |
| 3 | DESI prediction | 15% w(z) departure at z=0.8, 3.4% H(z) deviation |

## Building PDFs

```bash
pdflatex 01_thermodynamic_foundation.tex && pdflatex 01_thermodynamic_foundation.tex
pdflatex 02_black_hole_aperture.tex && pdflatex 02_black_hole_aperture.tex
pdflatex 03_cosmic_relaxation.tex && pdflatex 03_cosmic_relaxation.tex
```

## Citation

If you use this work, please cite:

```bibtex
@article{todd2026infodynamics,
  title={A Thermodynamic Foundation for the Second Law of Infodynamics},
  author={Todd, Ian},
  journal={IPI Letters},
  year={2026}
}

@article{todd2026aperture,
  title={Time as Information Rate Through Dimensional Apertures},
  author={Todd, Ian},
  journal={IPI Letters},
  year={2026}
}

@article{todd2026knot,
  title={Time, Mathematics, and the Relaxing Knot},
  author={Todd, Ian},
  journal={IPI Letters},
  year={2026}
}
```

## Related Work

This trilogy extends:
- Vopson & Lepadatu (2022), "Second law of information dynamics"
- Vopson (2023), "Second law of infodynamics and the simulated universe hypothesis"
- Jacobson (1995), "Thermodynamics of spacetime: the Einstein equation of state"

And connects to the author's broader research program on dimensional constraints in complex systems (see [BioSystems publications](https://coherencedynamics.com/papers)).

## License

MIT

## Author

Ian Todd
Sydney Medical School, University of Sydney
itod2305@uni.sydney.edu.au
