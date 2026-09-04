# Repository Guidelines

## Project Structure & Module Organization

This project models New York City ride-hailing dynamic pricing and fleet allocation. Prefer the supported modular pipeline in `src2/`:

- `src2/config.py` defines validated simulation and learning configuration.
- `src2/assets.py` prepares and loads demand, fare, and destination-model assets.
- `src2/algorithms.py`, `simulation.py`, and `experiments.py` contain learning agents, the market environment, and experiment loops.
- `src2/reporting.py` writes result CSVs and plots; `src2/run_experiments.py` is the supported CLI.
- `src/` contains legacy scripts; change it only when maintaining legacy workflows.
- `tests/` holds pytest tests. `archive/` is historical code and should not receive new features.

Prepared inputs and trained model artifacts belong in `data/`; generated runs belong in `experiments/`. Do not commit large raw trip data or regenerated outputs unless the task specifically requires it.

## Build, Test, and Development Commands

Use Python 3.10+ and a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests
```

Run a small supported experiment with `python -m src2.run_experiments --episodes 70 --steps 120`. Build or refresh assets from raw data with `python -m src2.run_experiments --prepare-assets --theta 0.4`. The legacy runner is `python src/experiment_pipeline.py --episodes 70 --steps 120`.

## Coding Style & Naming Conventions

Write Python with four-space indentation, type annotations for public functions, and concise docstrings. Follow the existing style: `snake_case` for functions, variables, and modules; `PascalCase` for classes; uppercase names for constants (for example, `PRICE_ACTIONS`). Keep configuration immutable through the existing frozen dataclasses and validate parameters at module boundaries. No formatter or linter is configured, so match nearby code and keep imports ordered: standard library, third-party packages, then local imports.

## Testing Guidelines

Use pytest and name test files `test_*.py` and test functions `test_*`. Add focused unit tests for algorithm or configuration changes; use small deterministic inputs and `numpy.random.default_rng(seed)` where randomness is involved. Tests that load prepared assets require the committed files in `data/`. Run `python -m pytest tests` before opening a pull request.

## Commit & Pull Request Guidelines

The available Git history contains only an initial commit, so no established commit convention exists. Use short imperative subjects, such as `Add fleet allocation validation`. Keep commits scoped. Pull requests should explain the simulation or data-assumption change, list commands run, link relevant issues, and include plots or CSV samples when reporting output changes.
