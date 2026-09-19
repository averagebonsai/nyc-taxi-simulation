"""Command-line entry point for the Python baseline implementation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Set before config imports NumPy so every ProcessPool worker remains a
# one-core session even when a cloud image defaults BLAS to many threads.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

if __package__:
    from .config import read_input
    from .reporting import write_eqm_input, write_eqm_plot, write_figure4_input, write_state_visit_log, write_training_visit_log
    from .simulation import run_experiment
else:  # Supports `python calvano-sourcecode/main.py` despite the requested hyphenated folder name.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from config import read_input
    from reporting import write_eqm_input, write_eqm_plot, write_figure4_input, write_state_visit_log, write_training_visit_log
    from simulation import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Calvano baseline Q-learning model.")
    parser.add_argument("--input", type=Path, required=True, help="Baseline A_InputParameters.txt file")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for A_irToBR.txt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--impulse-periods", type=int, default=15)
    parser.add_argument("--impulse-cycles", type=int, default=1, help="Number of repeated deviations by each agent")
    q_tables = parser.add_mutually_exclusive_group()
    q_tables.add_argument("--save-q-tables", type=Path, help="Empty directory for per-session trained Q-table archives")
    q_tables.add_argument("--load-q-tables", type=Path, help="Existing Q-table archive; skips Q-learning and runs post-training analysis")
    q_tables.add_argument("--resume-q-tables", type=Path, help="Partial Q-table archive; trains only missing session seeds")
    args = parser.parse_args()
    batch, experiments = read_input(args.input)
    for experiment in experiments:
        _, sessions, summary = run_experiment(
            batch,
            experiment,
            seed=args.seed,
            impulse_periods=args.impulse_periods,
            impulse_cycles=args.impulse_cycles,
            save_q_tables=args.save_q_tables,
            load_q_tables=args.load_q_tables,
            resume_q_tables=args.resume_q_tables,
        )
        visits = write_training_visit_log(
            args.output_dir / "A_trainingVisits.csv",
            summary["JointActionVisits"],
            summary["StateVisits"],
            memory=batch.memory,
        )
        state_visits = write_state_visit_log(
            args.output_dir / "A_trainingStateVisits.csv",
            summary["StateVisits"],
            num_agents=batch.num_agents,
            num_prices=batch.num_prices,
            memory=batch.memory,
        )
        equilibrium_data = write_eqm_input(args.output_dir, summary, args.impulse_cycles)
        equilibrium_plot = write_eqm_plot(equilibrium_data)
        if batch.impulse_response_to_br:
            output = write_figure4_input(args.output_dir / "A_irToBR.txt", summary)
            print(
                f"experiment {experiment.identifier}: {sum(item.converged for item in sessions)}/{len(sessions)} "
                f"sessions converged; wrote {output}, {visits}, {state_visits}, {equilibrium_data}, and {equilibrium_plot}"
            )


if __name__ == "__main__":
    main()
