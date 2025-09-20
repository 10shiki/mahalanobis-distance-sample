# Repository Guidelines

## Project Structure & Module Organization
- `m_distance.py` holds the Mahalanobis distance demo, including math helpers, CLI entry point, and optional plotting support.
- Add reusable utilities as standalone functions within `m_distance.py` or extract to `metrics/` if the module grows; place visual helpers in `visualization/` to keep compute logic lean.
- Store datasets or notebooks in `data/` and `notebooks/` (create as needed) so that the root remains focused on Python modules.

## Environment & Tooling
- Developed against Python 3.11; use a virtual environment (`python -m venv .venv && source .venv/bin/activate`).
- Only standard library is required for core functionality; `matplotlib` is optional for plotting (`pip install matplotlib`).

## Build, Test, and Development Commands
- `python m_distance.py` runs the demo, prints mean/covariance tables, and launches the visualization when `matplotlib` is available.
- `python -m compileall m_distance.py` is a quick syntax check for CI or pre-commit hooks.
- `pytest` is the preferred test runner once tests are added (see below). Configure `PYTHONPATH=.` if tests import the root module.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and descriptive, snake_case function names; type hints are required for public helpers.
- Keep numerical helpers pure and side-effect free; isolate plotting behind guard clauses as in `plot_data`.
- Document non-obvious math in docstrings and use concise comments for numerical safeguards or derivations.

## Testing Guidelines
- Place tests under `tests/` mirroring module names (e.g., `tests/test_m_distance.py`).
- Use `pytest` with parametrized cases for statistical helpers; include edge cases for malformed input (dimension mismatch, singular matrices).
- Aim for >90% coverage on computational functions; mark visualization branches with `# pragma: no cover` only when interaction is unavoidable.

## Commit & Pull Request Guidelines
- Use conventional commits (`feat:`, `fix:`, `refactor:`) with imperative summaries under 60 characters and context details in the body.
- Reference issue IDs when available (`Refs #12`) and describe numerical validations or plots generated during testing.
- Pull requests should list executed commands, attach resulting console logs or images, and flag dependencies (`matplotlib`) so reviewers can reproduce results.
