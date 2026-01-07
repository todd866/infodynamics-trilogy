# IPI Letters Trilogy

Three papers extending Vopson's infodynamics framework, scaling from microscopic to cosmological.

## The Arc

| Paper | Scale | Core Claim | File |
|-------|-------|------------|------|
| 1. Thermodynamic Foundation | Microscopic | Asymmetric states cost work; structure-information decays toward symmetry | `01_thermodynamic_foundation.tex` |
| 2. Black Hole Aperture | Mesoscopic | Time = information rate through dimensional apertures; horizons are where channels close | `02_black_hole_aperture.tex` |
| 3. Cosmic Relaxation | Cosmological | Universe is a relaxing knot; time, math, and physics emerge from constraint dynamics | `03_cosmic_relaxation.tex` |

## Status

- **Paper 1**: Accepted with minor revisions (addressed). Ready.
- **Paper 2**: Complete draft with simulations. May benefit from tightening.
- **Paper 3**: Complete draft. Most ambitious/speculative. Consider focusing.

## Cross-References

- Paper 2 cites Paper 1 (`\cite{todd2025infodynamics}`)
- Paper 3 cites both Papers 1 and 2, explicitly frames itself as "third in a trilogy"

## What Makes This a Trilogy

1. **Scale progression**: observer → spacetime → cosmos
2. **Shared formalism**: participation ratio, KL divergence, Landauer bounds
3. **Consistent ontology**: high-dimensional geometry is fundamental; information is what observers can commit about it
4. **Extends Vopson**: provides thermodynamic grounding for second law of infodynamics

## Polish Priorities

### Paper 2 (Black Hole Aperture)
- Section 4.3 (LIGO/ringdown) may be overreach - we're not modeling gravitational waves
- Consider cutting or significantly trimming
- Core insight (aperture → time dilation) is strong; don't dilute it

### Paper 3 (Cosmic Relaxation)
- Two big ideas competing: (1) universe as knot, (2) mathematics as contingent
- The math-contingency section (§5) could be its own paper
- Consider focusing on cosmology + infodynamics for this trilogy
- Dark energy predictions are the strongest testable claim

### Both Papers 2 & 3
- Check that figures compile correctly from this folder
- Ensure cross-references to Paper 1 use consistent citation key
- Tone should match Paper 1 (accessible, grounded, not overclaiming)

## Building PDFs

```bash
cd /Users/iantodd/Projects/highdimensional/physics/IPI_trilogy
pdflatex 01_thermodynamic_foundation.tex
pdflatex 02_black_hole_aperture.tex
pdflatex 03_cosmic_relaxation.tex
```

## Target

IPI Letters (Vopson's journal). Relationship is warm - he engaged with Ian early on. Goal is to become a contributor to the infodynamics research program, not just a one-off author.

## Quality Bar

These should be papers we'd be proud of regardless of venue. IPI Letters may be niche, but "niche and good" beats "niche and sloppy."
