# Contributing

Thank you for considering a contribution.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Check

Before submitting changes, run the smoke check from `README.md` and, if Node.js
is available:

```bash
node --check web/app.js
```

## Contribution Guidelines

- Keep calculation input data in `data/`.
- Keep calculation code in `src/`.
- Keep the web explorer self-contained in `web/`.
- Do not commit generated caches such as `__pycache__/`, `.pyc`, `.DS_Store`,
  local virtual environments, or local output files.
- Document any change to matrix elements, residuals, benchmark values, or
  physical conventions in `DATA_SOURCES.md` and `README.md`.

## Scientific Changes

Changes to physics inputs should include:

- the source reference,
- the affected state and transition,
- the unit convention,
- whether the value is an input, residual, or benchmark,
- a before/after smoke-check result when applicable.
