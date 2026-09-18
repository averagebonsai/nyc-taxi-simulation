"""Command-line entry point for the Python baseline implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .config import read_input
    from .reporting import write_figure4_input
    from .simulation import run_experiment
else:  # Supports `python calvano-sourcecode/main.py` despite the requested hyphenated folder name.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from config import read_input
    from reporting import write_figure4_input
    from simulation import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Calvano baseline Q-learning model.")
    parser.add_argument("--input", type=Path, required=True, help="Baseline A_InputParameters.txt file")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for A_irToBR.txt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--impulse-periods", type=int, default=15)
    args = parser.parse_args()
    batch, experiments = read_input(args.input)
    for experiment in experiments:
        _, sessions, summary = run_experiment(batch, experiment, seed=args.seed, impulse_periods=args.impulse_periods)
        if batch.impulse_response_to_br:
            output = write_figure4_input(args.output_dir / "A_irToBR.txt", summary)
            print(f"experiment {experiment.identifier}: {sum(item.converged for item in sessions)}/{len(sessions)} sessions converged; wrote {output}")


if __name__ == "__main__":
    main()
