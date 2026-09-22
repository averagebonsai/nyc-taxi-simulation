"""Generate all requested Calvano impulse-response reports from saved Q-tables.

This command is independent of the memory setting: it reloads the learned
policy/state from each archive and performs only post-training analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    deviation_grid_steps: int | None = None,
    deviation_periods: int = 1,
) -> dict[str, np.ndarray | float]:
    """Run only the post-training analysis for one cycle/period specification."""
    _, _, summary = run_experiment(
        batch,
        experiment,
        impulse_periods=periods,
        impulse_cycles=cycles,
        load_q_tables=archive,
        deviation_grid_steps=deviation_grid_steps,
        deviation_periods=deviation_periods,
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
    *,
    state_representation: str = "joint",
    deviation_grid_steps: int | None = None,
    deviation_periods: int = 1,
    include_20_period_reports: bool = True,
    impulse_periods: int = 15,
) -> None:
    """Generate the one-, ten-, and fifty-cycle charts and their source CSVs.

    The function does not train. The supplied archive must have been trained
    with the supplied input file's state space, action grid, and learning
    parameters.
    """
    batch, experiments = read_input(input_path)
    batch = replace(
        batch,
        state_representation=state_representation,
    )
    if len(experiments) != 1:
        raise ValueError("Report generation currently supports one experiment row")
    experiment = experiments[0]
    archive = Path(q_tables)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    nash_price = float(np.mean(experiment.nash_prices))
    monopoly_price = float(np.mean(experiment.cooperative_prices))

    one_cycle = _summary(batch, experiment, archive, periods=impulse_periods, cycles=1, deviation_grid_steps=deviation_grid_steps, deviation_periods=deviation_periods)
    write_figure4_input(output / "A_irToBR_1_cycle.txt", one_cycle)
    one_cycle_data = write_impulse_response_csv(
        output / "1_cycle_impulse.csv", one_cycle, nash_price, monopoly_price
    )
    write_impulse_response_plots(
        one_cycle_data,
        "Prices of 2 Agents after Small Unilateral Deviation" if deviation_grid_steps else "Prices of 2 Agents after Unilateral Deviation",
        output / "figure_4_replicated",
    )

    ten_cycles_15 = _summary(batch, experiment, archive, periods=impulse_periods, cycles=10, deviation_grid_steps=deviation_grid_steps, deviation_periods=deviation_periods)
    write_figure4_input(output / f"A_irToBR_10_cycles_{impulse_periods}_episodes.txt", ten_cycles_15)
    ten_cycles_data = write_impulse_response_csv(
        output / f"10_cycles_{impulse_periods}_episodes.csv", ten_cycles_15, nash_price, monopoly_price
    )
    write_impulse_response_plots(
        ten_cycles_data,
        "Prices of 2 Agents after 10 Cycles of Small Unilateral Deviation" if deviation_grid_steps else "Prices of 2 Agents after 10 Cycles of Unilateral Deviation",
        output / "10_cycles",
    )
    _write_equilibrium_plot(output, ten_cycles_15, 10, impulse_periods, nash_price, monopoly_price)

    fifty_cycles_15 = _summary(batch, experiment, archive, periods=impulse_periods, cycles=50, deviation_grid_steps=deviation_grid_steps, deviation_periods=deviation_periods)
    _write_equilibrium_plot(output, fifty_cycles_15, 50, impulse_periods, nash_price, monopoly_price)

    if include_20_period_reports:
        ten_cycles_20 = _summary(batch, experiment, archive, periods=20, cycles=10, deviation_grid_steps=deviation_grid_steps, deviation_periods=deviation_periods)
        _write_equilibrium_plot(output, ten_cycles_20, 10, 20, nash_price, monopoly_price)

        fifty_cycles_20 = _summary(batch, experiment, archive, periods=20, cycles=50, deviation_grid_steps=deviation_grid_steps, deviation_periods=deviation_periods)
        _write_equilibrium_plot(output, fifty_cycles_20, 50, 20, nash_price, monopoly_price)


def generate_endpoint_report(
    input_path: str | Path,
    q_tables: str | Path,
    output_directory: str | Path,
    *,
    cycles: int,
    periods: int,
    state_representation: str = "joint",
    deviation_grid_steps: int | None = None,
    deviation_periods: int = 1,
) -> Path:
    """Generate one equilibrium-cycle chart and CSV without retraining."""
    batch, experiments = read_input(input_path)
    batch = replace(
        batch,
        state_representation=state_representation,
    )
    experiment = experiments[0]
    output = Path(output_directory)
    nash_price = float(np.mean(experiment.nash_prices))
    monopoly_price = float(np.mean(experiment.cooperative_prices))
    summary = _summary(batch, experiment, Path(q_tables), periods, cycles, deviation_grid_steps, deviation_periods)
    return _write_equilibrium_plot(output, summary, cycles, periods, nash_price, monopoly_price)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Calvano reports from saved Q-tables.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--q-tables", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-representation", choices=("joint", "opponent"), default="joint")
    parser.add_argument("--deviation-grid-steps", type=int)
    parser.add_argument("--deviation-periods", type=int, default=1, help="Static-best-response periods at the start of each cycle")
    parser.add_argument("--endpoint-only", action="store_true")
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--periods", type=int)
    parser.add_argument("--standard-15-only", action="store_true", help="Omit the supplementary 20-period endpoint charts.")
    args = parser.parse_args()
    if args.endpoint_only:
        if args.cycles is None or args.periods is None:
            parser.error("--endpoint-only requires --cycles and --periods")
        generate_endpoint_report(args.input, args.q_tables, args.output_dir, cycles=args.cycles, periods=args.periods, state_representation=args.state_representation, deviation_grid_steps=args.deviation_grid_steps, deviation_periods=args.deviation_periods)
    else:
        generate_all_reports(
            args.input,
            args.q_tables,
            args.output_dir,
            state_representation=args.state_representation,
            deviation_grid_steps=args.deviation_grid_steps,
            deviation_periods=args.deviation_periods,
            include_20_period_reports=not args.standard_15_only,
            impulse_periods=args.periods or 15,
        )


if __name__ == "__main__":
    main()
