# Sr-87 515 nm Polarizability Toolkit

Open-source calculation and visualization package for the dynamic polarizability
of fermionic strontium-87 near 515 nm.

The workflow is:

1. Compute electronic Sr I / Sr-88-like polarizability components
   `alpha0`, `alpha1`, and `alpha2` from a sum-over-states model.
2. Project the electronic `mJ`-resolved polarizability to Sr-87 hyperfine
   states `F,mF` using Clebsch-Gordan weights for nuclear spin `I=9/2`.
3. Use the resulting polarizabilities to estimate light shifts, scattering,
   low-field Zeeman shifts, and magic-wavelength candidates.

## Repository Layout

```text
data/        Input CSV data needed by the calculation
src/         Python calculation code
web/         Standalone English magic-wavelength explorer
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On the original development machine, the verified interpreter was:

```bash
/opt/anaconda3/bin/python3
```

## Quick Start

Compute Sr-87 `5s5p 3P1`, `F=11/2`, `mF=9/2` at `515.2 nm` for linear
polarization along the quantization axis:

```bash
python src/sr87_polarizability.py \
  --state-label "5s5p 3P1" \
  --wavelength-nm 515.2 \
  --geometry linear_z \
  --F 11/2 \
  --mF 9/2
```

Include a Gaussian beam intensity, first-pass scattering/heating estimate, and
a low-field Zeeman estimate:

```bash
python src/sr87_polarizability.py \
  --state-label "5s5p 3P1" \
  --wavelength-nm 515.2 \
  --geometry linear_xz \
  --theta-deg 0 \
  --F 11/2 \
  --mF 9/2 \
  --power-W 0.02 \
  --waist-um 0.8 \
  --transmission 0.75 \
  --include-scattering \
  --magnetic-field-G 1
```

Write a machine-readable JSON result:

```bash
python src/sr87_polarizability.py \
  --state-label "5s2 1S0" \
  --wavelength-nm 515.2 \
  --geometry linear_z \
  --output-json result_1s0_515p2.json
```

## Input Data

The calculation reads input files from `data/` by default:

| File | Role |
| --- | --- |
| `levels_sr1.csv` | Sr I energy levels, parity, `J`, and energies. |
| `lines_sr1.csv` | Transition-line and `Aki` data for fallback RDME conversion. |
| `literature_matrix_elements_sr1.csv` | Preferred reduced dipole matrix elements. |
| `residual_contributions_sr1.csv` | Component-level residual, core, and tail corrections. |
| `benchmarks_sr1.csv` | Literature benchmark values used for validation. |
| `sr_known_transitions.csv` | Transition metadata for scattering/heating diagnostics. |

Set `SR87_POL_DATA_DIR` to use a different data directory:

```bash
export SR87_POL_DATA_DIR=/path/to/data
```

## Model Summary

For each electronic state, the code builds an E1 channel catalog and evaluates
sum-over-states contributions:

```text
alpha0 ~ sum Delta |D|^2 / (Delta^2 - omega^2)
alpha1 ~ sum C1 * 2 omega |D|^2 / (Delta^2 - omega^2)
alpha2 ~ sum C2 * Delta |D|^2 / (Delta^2 - omega^2)
```

where `D` is a reduced dipole matrix element and `C1`, `C2` are angular factors
containing Wigner 6-j symbols. Residual terms are added at the electronic
component level.

The Sr-87 projection is then:

```text
alpha(F,mF) = sum_mJ |CG(J,mJ; I,mI -> F,mF)|^2 alpha(mJ)
```

with `I=9/2`.

## Magic Wavelength Web Explorer

The static web app is in `web/`.

```bash
cd web
python -m http.server 8022
```

Then open:

```text
http://127.0.0.1:8022/
```

The browser reads `web/data/magic_web_data.json` and computes

```text
Delta alpha(lambda) = alpha_excited(F',mF') - alpha_ground(F=9/2,mF)
```

Magic-wavelength candidates are roots of `Delta alpha(lambda)=0`. The web app
supports `Delta mF = 0, +1, -1` channels.

## Verification

Run this smoke check from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import sys
sys.path.insert(0, "src")
import sr87_polarizability as sr87
from sympy import Rational

r = sr87.compute_sr87_polarizability(
    state_label="5s5p 3P1",
    wavelength_nm=515.2,
    geometry_name="linear_z",
    F_value=Rational(11, 2),
    mF_value=Rational(9, 2),
)

print(round(r["electronic_components_au"]["scalar"], 12))
print(round(r["electronic_components_au"]["vector"], 12))
print(round(r["electronic_components_au"]["tensor"], 12))
print(r["hyperfine_results"][0]["F"], r["hyperfine_results"][0]["mF"], round(r["hyperfine_results"][0]["alpha_au"], 12))
PY
```

Expected output:

```text
769.15134011357
-9.051625123548
-110.561487870998
11/2 9/2 718.896118354025
```

Check the web JavaScript syntax if Node.js is available:

```bash
node --check web/app.js
```

## Limitations

- `3P1 alpha1` is a cancellation-sensitive quantity and remains sensitive to
  matrix-element and residual conventions.
- `3P2` is strengthened with Delaware/Trautmann data, but it should still be
  treated cautiously for experimental operating-point selection.
- The scattering/heating model is a first-pass scalar E1 estimate, not a full
  optical Bloch or branching-ratio model.
- The Zeeman module is a low-field first-order engineering estimate.

## Citation

If you use this package, please cite the repository and the source literature
listed in `DATA_SOURCES.md`. A machine-readable citation file is provided as
`CITATION.cff`.
