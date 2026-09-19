"""Train/resume one Calvano experiment, then generate every requested report."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import read_input
from generate_reports import generate_all_reports
from reporting import write_figure4_input, write_state_visit_log, write_training_visit_log
from simulation import run_experiment


def run_all_reports(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    seed: int = 1,
    q_tables: str | Path | None = None,
) -> None:
    """Train or resume, checkpoint sessions, and render the complete report set.

    A new archive is created when absent. If it already exists, only its missing
    session seeds are trained; a complete archive is simply reloaded.
    """
    input_file = Path(input_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    archive = Path(q_tables) if q_tables is not None else output / "q_tables"
    batch, experiments = read_input(input_file)

    for experiment in experiments:
        if archive.exists():
            _, sessions, summary = run_experiment(
                batch,
                experiment,
                seed=seed,
                impulse_periods=15,
                impulse_cycles=1,
                resume_q_tables=archive,
            )
        else:
            _, sessions, summary = run_experiment(
                batch,
                experiment,
                seed=seed,
                impulse_periods=15,
                impulse_cycles=1,
                save_q_tables=archive,
            )
        write_training_visit_log(
            output / "A_trainingVisits.csv",
            summary["JointActionVisits"],
            summary["StateVisits"],
            memory=batch.memory,
        )
        write_state_visit_log(
            output / "A_trainingStateVisits.csv",
            summary["StateVisits"],
            num_agents=batch.num_agents,
            num_prices=batch.num_prices,
            memory=batch.memory,
        )
        write_figure4_input(output / "A_irToBR.txt", summary)
        print(f"experiment {experiment.identifier}: {sum(item.converged for item in sessions)}/{len(sessions)} sessions converged")

    generate_all_reports(input_file, archive, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/resume Calvano and generate all reports in one command.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--q-tables", type=Path, help="Checkpoint archive; defaults to OUTPUT_DIR/q_tables")
    args = parser.parse_args()
    run_all_reports(args.input, args.output_dir, seed=args.seed, q_tables=args.q_tables)


if __name__ == "__main__":
    main()
