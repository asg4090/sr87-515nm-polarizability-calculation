# Data Sources

This file summarizes how the bundled data files are used by the calculation.

## Active Input Files

| File | Role |
| --- | --- |
| `data/levels_sr1.csv` | Sr I energy levels, term labels, `J`, parity, and energies. |
| `data/lines_sr1.csv` | Transition-line data used to infer fallback reduced dipole matrix elements from Einstein A coefficients. |
| `data/literature_matrix_elements_sr1.csv` | Preferred reduced dipole matrix elements used directly in the SOS sums. |
| `data/residual_contributions_sr1.csv` | Core, tail, and other residual corrections added after explicit SOS channels. |
| `data/benchmarks_sr1.csv` | Literature/reference targets used for validation only. |
| `data/sr_known_transitions.csv` | Transition metadata used by diagnostic and scattering routines. |

## Source Families

| Source family | Data form | How it is used |
| --- | --- | --- |
| Cooper PRX 2018 | RDME, residuals, 515.2 nm benchmark components | Main `1S0` and `3P1` 515 nm data and benchmark reference. |
| Delaware Atom Portal | RDME values | Main `3P2` matrix-element source in this release snapshot. |
| Trautmann PRR 2023 | `3P2` residual and benchmark values | Scalar core residual and 1064 nm validation reference. |
| Kestler PRA 2022 | `3P1` 473 nm benchmark values | External benchmark only. |
| Madjarov thesis | 813 nm and selected `3P2` reference values | Benchmark/reference only. |
| NIST ASD / Sansonetti-Nave Sr I | A coefficients and transition metadata | Fallback channel data when no higher-priority RDME is available. |
| Sr1Pol | Reference electronic polarizability curves | Used to generate the bundled web comparison data; not an SOS input. |

## Data Priority

1. Literature RDME values are preferred when available.
2. The recommended `3P2` snapshot uses Delaware RDME values.
3. Trautmann `3P2` scalar core residual is enabled in the recommended scenario.
4. A-coefficient conversion is used as a fallback for channels without preferred literature RDME values.
5. Benchmark/reference values are used for validation only.
6. Optional residuals are disabled unless `--include-optional-residuals` is passed.

## Important Convention

Residual rows are additive component-level corrections:

```text
alpha_component_total = alpha_component_SOS + residual_component
```

They are applied before the Sr-87 hyperfine projection.
