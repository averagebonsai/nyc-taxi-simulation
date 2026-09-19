"""Generate all requested Calvano impulse-response reports from saved Q-tables.

This command is independent of the memory setting: it reloads the learned
policy/state from each archive and performs only post-training analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import BatchConfig, ExperimentConfig, read_input
from reporting import (
    impulse_y_limits,
    write_eqm_input,
    write_eqm_plot,
    write_figure4_input,
    write_impulse_response_csv,
    write_impulse_response_plots,
)
from simulation import run_experiment


def _summary(
    batch: BatchConfig,
    experiment: ExperimentConfig,
    archive: Path,
    periods: int,
    cycles: int,
) -> dict[str, np.ndarray | float]:
    """Run only the post-training analysis for one cycle/period specification."""
    _, _, summary = run_experiment(
        batch,
        experiment,
        impulse_periods=periods,
        impulse_cycles=cycles,
        load_q_tables=archive,
    )
    return summary


def _write_equilibrium_plot(
    output_directory: Path,
    summary: dict[str, np.ndarray | float],
    cycles: int,
    periods: int,
    nash_price: float,
    monopoly_price: float,
) -> Path:
    data = write_eqm_input(output_directory, summary, cycles)
    return write_eqm_plot(
        data,
        tick_interval=1 if cycles <= 10 else 5,
        plot_style="lines",
        y_limits=impulse_y_limits(summary, nash_price, monopoly_price),
        title=f"Prices of Agents Prior to Each Deviation ({periods} Episodes, {cycles} Cycles)",
        reference_prices=(nash_price, monopoly_price),
        show_legend=True,
    )


def generate_all_reports(
    input_path: str | Path,
    q_tables: str | Path,
    output_directory: str | Path,
) -> None:
    """Generate the one-, ten-, and fifty-cycle charts and their source CSVs.

    The function does not train. The supplied archive must have been trained
    with the supplied input file's state space, action grid, and learning
    parameters.
    """
    batch, experiments = read_input(input_path)
    if len(experiments) != 1:
        raise ValueError("Report generation currently supports one experiment row")
    experiment = experiments[0]
    archive = Path(q_tables)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    nash_price = float(np.mean(experiment.nash_prices))
    monopoly_price = float(np.mean(experiment.cooperative_prices))

    one_cycle = _summary(batch, experiment, archive, periods=15, cycles=1)
    write_figure4_input(output / "A_irToBR_1_cycle.txt", one_cycle)
    one_cycle_data = write_impulse_response_csv(
        output / "1_cycle_impulse.csv", one_cycle, nash_price, monopoly_price
    )
    write_impulse_response_plots(
        one_cycle_data,
        "Prices of 2 Agents after Unilateral Deviation",
        output / "figure_4_replicated",
    )

    ten_cycles_15 = _summary(batch, experiment, archive, periods=15, cycles=10)
    write_figure4_input(output / "A_irToBR_10_cycles_15_episodes.txt", ten_cycles_15)
    ten_cycles_data = write_impulse_response_csv(
        output / "10_cycles_15_episodes.csv", ten_cycles_15, nash_price, monopoly_price
    )
    write_impulse_response_plots(
        ten_cycles_data,
        "Prices of 2 Agents after 10 Cycles of Unilateral Deviation",
        output / "10_cycles",
    )
    _write_equilibrium_plot(output, ten_cycles_15, 10, 15, nash_price, monopoly_price)

    ten_cycles_20 = _summary(batch, experiment, archive, periods=20, cycles=10)
    _write_equilibrium_plot(output, ten_cycles_20, 10, 20, nash_price, monopoly_price)

    fifty_cycles_15 = _summary(batch, experiment, archive, periods=15, cycles=50)
    _write_equilibrium_plot(output, fifty_cycles_15, 50, 15, nash_price, monopoly_price)

    fifty_cycles_20 = _summary(batch, experiment, archive, periods=20, cycles=50)
    _write_equilibrium_plot(output, fifty_cycles_20, 50, 20, nash_price, monopoly_price)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Calvano reports from saved Q-tables.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--q-tables", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate_all_reports(args.input, args.q_tables, args.output_dir)


if __name__ == "__main__":
    main()
