"""Writers compatible with the supplied Figure 4 R script."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def write_figure4_input(path: str | Path, summary: dict[str, np.ndarray | float]) -> Path:
    """Write the whitespace-delimited ``A_irToBR.txt`` consumed by Figure 4."""
    target = Path(path)
    dev = np.asarray(summary["AggrDevPriceShock"], dtype=float)
    non_dev = np.asarray(summary["AggrNonDevPriceShock"], dtype=float)
    fields = ["AggrPricePre"]
    fields.extend(f"AggrDevPriceShockPer{period:03d}" for period in range(1, len(dev) + 1))
    fields.extend(f"AggrNonDevPriceShockPer{period:03d}" for period in range(1, len(non_dev) + 1))
    values = [float(summary["AggrPricePre"]), *dev.tolist(), *non_dev.tolist()]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(" ".join(fields) + "\n" + " ".join(f"{value:.7f}" for value in values) + "\n", encoding="utf-8")
    return target


def impulse_y_limits(
    summary: dict[str, np.ndarray | float],
    nash_price: float,
    monopoly_price: float,
) -> tuple[float, float]:
    """Return the Figure 4 range, including both equilibrium reference prices."""
    deviator_prices = np.asarray(summary["AggrDevPriceShock"], dtype=float)
    rival_prices = np.asarray(summary["AggrNonDevPriceShock"], dtype=float)
    lower = min(float(deviator_prices.min()), float(rival_prices.min()), nash_price)
    upper = max(float(deviator_prices.max()), float(rival_prices.max()), monopoly_price)
    # A small margin keeps equilibrium reference lines visible instead of
    # clipping them against the plot boundary.
    margin = max((upper - lower) * 0.02, 1e-6)
    return lower - margin, upper + margin


def write_impulse_response_csv(
    path: str | Path,
    summary: dict[str, np.ndarray | float],
    nash_price: float,
    monopoly_price: float,
) -> Path:
    """Write the complete impulse-response path so it can be re-plotted later."""
    deviator_prices = np.asarray(summary["AggrDevPriceShock"], dtype=float)
    rival_prices = np.asarray(summary["AggrNonDevPriceShock"], dtype=float)
    if len(deviator_prices) != len(rival_prices):
        raise ValueError("Impulse-response series must have equal lengths")
    y_min, y_max = impulse_y_limits(summary, nash_price, monopoly_price)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["period", "deviator_price", "rival_price", "nash_price", "monopoly_price", "y_min", "y_max"])
        writer.writerow([0, f"{float(summary['AggrPricePre']):.7f}", f"{float(summary['AggrPricePre']):.7f}", f"{nash_price:.7f}", f"{monopoly_price:.7f}", f"{y_min:.7f}", f"{y_max:.7f}"])
        for period, (deviator_price, rival_price) in enumerate(zip(deviator_prices, rival_prices), start=1):
            writer.writerow([period, f"{deviator_price:.7f}", f"{rival_price:.7f}", f"{nash_price:.7f}", f"{monopoly_price:.7f}", f"{y_min:.7f}", f"{y_max:.7f}"])
    return target


def write_impulse_response_plot(
    path: str | Path,
    title: str,
    output_path: str | Path,
    *,
    show_legend: bool = True,
) -> Path:
    """Render a Figure 4-style impulse response from its CSV data."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{source} contains no impulse-response observations")
    periods = np.asarray([int(row["period"]) for row in rows])
    deviator_prices = np.asarray([float(row["deviator_price"]) for row in rows])
    rival_prices = np.asarray([float(row["rival_price"]) for row in rows])
    nash_price = float(rows[0]["nash_price"])
    monopoly_price = float(rows[0]["monopoly_price"])
    y_min, y_max = float(rows[0]["y_min"]), float(rows[0]["y_max"])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(periods, deviator_prices, color="black", marker="o", markersize=3.5, linewidth=1.8, label="Deviating agent")
    axis.plot(periods, rival_prices, color="gray", marker="^", markersize=3.8, linewidth=1.8, linestyle="--", label="Non-deviating agent")
    axis.axhline(nash_price, color="dimgray", linestyle=":", linewidth=1.0, label="Nash equilibrium")
    axis.axhline(monopoly_price, color="dimgray", linestyle=(0, (4, 2)), linewidth=1.0, label="Monopoly price")
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Time")
    axis.set_ylabel("Price")
    axis.set_ylim(y_min, y_max)
    ticks = sorted(set([0, 1, *range(5, int(periods.max()) + 1, 5), int(periods.max())]))
    axis.set_xticks(ticks)
    axis.grid(axis="y", color="0.9", linewidth=0.8)
    if show_legend:
        axis.legend(loc="best", fontsize=7)
    figure.tight_layout()
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def write_impulse_response_plots(
    path: str | Path,
    title: str,
    output_stem: str | Path,
    *,
    show_legend: bool = True,
) -> tuple[Path, Path]:
    """Write matching PDF and PNG impulse-response charts from one CSV."""
    stem = Path(output_stem)
    if stem.suffix:
        stem = stem.with_suffix("")
    pdf = write_impulse_response_plot(path, title, stem.with_suffix(".pdf"), show_legend=show_legend)
    png = write_impulse_response_plot(path, title, stem.with_suffix(".png"), show_legend=show_legend)
    return pdf, png


def write_eqm_input(
    path: str | Path,
    summary: dict[str, np.ndarray | float],
    impulse_cycles: int,
) -> Path:
    """Write cycle-end equilibrium data to CSV and return its path.

    Point zero is the pre-deviation price. Each later point samples the last
    learned-policy period in a cycle, immediately before the same agent receives
    the next forced deviation. The two plotted series are the deviator and rival
    roles, averaged by the impulse-response analysis across the symmetric firms.
    """
    if impulse_cycles < 1:
        raise ValueError("impulse_cycles must be at least 1")
    deviator_prices = np.asarray(summary["AggrDevPriceShock"], dtype=float)
    rival_prices = np.asarray(summary["AggrNonDevPriceShock"], dtype=float)
    if len(deviator_prices) != len(rival_prices) or len(deviator_prices) % impulse_cycles:
        raise ValueError("Impulse-response observations must divide evenly into impulse_cycles")

    periods_per_cycle = len(deviator_prices) // impulse_cycles
    cycle_end_indices = np.arange(periods_per_cycle - 1, len(deviator_prices), periods_per_cycle)
    time = np.arange(impulse_cycles + 1)
    pre_deviation_price = float(summary["AggrPricePre"])
    deviator_points = np.concatenate(([pre_deviation_price], deviator_prices[cycle_end_indices]))
    rival_points = np.concatenate(([pre_deviation_price], rival_prices[cycle_end_indices]))

    output_directory = Path(path)
    if output_directory.suffix:
        output_directory = output_directory.parent
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{impulse_cycles}_eqm"
    if periods_per_cycle != 15:
        stem += f"_{periods_per_cycle}_periods"
    target = output_directory / f"{stem}.csv"

    # These match figure_4.R: retain the original reference prices while also
    # accommodating an impulse response that extends beyond those references.
    y_min = min(float(deviator_prices.min()), float(rival_prices.min()), 1.47293)
    y_max = max(float(deviator_prices.max()), float(rival_prices.max()), 1.92498)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cycle", "deviator_price", "rival_price", "y_min", "y_max"])
        for cycle, deviator_price, rival_price in zip(time, deviator_points, rival_points):
            writer.writerow([cycle, f"{deviator_price:.7f}", f"{rival_price:.7f}", f"{y_min:.7f}", f"{y_max:.7f}"])
    return target


def write_eqm_plot(
    path: str | Path,
    tick_interval: int = 5,
    plot_style: str = "lines",
    y_limits: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
    reference_prices: tuple[float, float] | None = None,
    show_legend: bool = False,
) -> Path:
    """Render an equilibrium PNG from a CSV written by :func:`write_eqm_input`.

    ``plot_style`` is either ``"lines"`` or ``"points"``.  By default the
    Figure 4-compatible limits embedded in the CSV are used; callers can pass
    ``y_limits`` to inspect the small variation around the cycle endpoints.
    """
    if tick_interval < 1:
        raise ValueError("tick_interval must be at least 1")
    if plot_style not in {"lines", "points"}:
        raise ValueError("plot_style must be 'lines' or 'points'")
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{source} contains no equilibrium observations")
    time = np.asarray([int(row["cycle"]) for row in rows])
    deviator_points = np.asarray([float(row["deviator_price"]) for row in rows])
    rival_points = np.asarray([float(row["rival_price"]) for row in rows])
    y_min, y_max = float(rows[0]["y_min"]), float(rows[0]["y_max"])
    target = Path(output_path) if output_path is not None else source.with_suffix(".png")
    if y_limits is not None:
        y_min, y_max = y_limits
        if y_min >= y_max:
            raise ValueError("y_limits must have a lower bound below its upper bound")

    figure, axis = plt.subplots(figsize=(8, 4))
    if plot_style == "lines":
        axis.plot(time, deviator_points, color="black", linewidth=1.5, label="Deviating agent")
        axis.plot(time, rival_points, color="gray", linewidth=1.5, linestyle="--", label="Non-deviating agent")
    else:
        axis.scatter(time, deviator_points, color="black", marker="o", s=34)
        axis.scatter(time, rival_points, color="gray", marker="^", s=38)
    if reference_prices is not None:
        nash_price, monopoly_price = reference_prices
        axis.axhline(nash_price, color="dimgray", linestyle=":", linewidth=1.0, label="Nash equilibrium")
        axis.axhline(monopoly_price, color="dimgray", linestyle=(0, (4, 2)), linewidth=1.0, label="Monopoly price")
    if title is not None:
        axis.set_title(title, fontsize=10)
    axis.set_xlabel("Deviation cycle")
    axis.set_ylabel("Price")
    axis.set_xticks(np.arange(0, time.max() + 1, tick_interval))
    axis.set_ylim(y_min, y_max)
    axis.grid(axis="y", color="0.9", linewidth=0.8)
    if show_legend:
        axis.legend(loc="best", fontsize=7)
    figure.tight_layout()
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def write_training_visit_log(
    path: str | Path,
    joint_action_visits: np.ndarray,
    state_visits: np.ndarray,
    *,
    memory: int = 1,
) -> Path:
    """Write aggregate counts for each joint action.

    For one-period memory this keeps the original combined action/state report.
    Longer memories have a distinct state space, written by
    :func:`write_state_visit_log`.
    """
    action_counts = np.asarray(joint_action_visits, dtype=np.int64)
    state_counts = np.asarray(state_visits, dtype=np.int64)
    if action_counts.ndim != 2:
        raise ValueError("The training visit log requires two agents")
    if memory < 0:
        raise ValueError("The training visit log cannot use negative memory")
    if memory == 1 and state_counts.size != action_counts.size:
        raise ValueError("One-period state visits must have one entry per joint action")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["agent_1_price_index", "agent_2_price_index", "joint_action_visits"]
        if memory == 1:
            header.append("state_visits")
        writer.writerow(header)
        for first_price in range(action_counts.shape[0]):
            for second_price in range(action_counts.shape[1]):
                row: list[int] = [first_price + 1, second_price + 1, int(action_counts[first_price, second_price])]
                if memory == 1:
                    state_index = first_price * action_counts.shape[1] + second_price
                    row.append(int(state_counts[state_index]))
                writer.writerow(row)
    return target


def write_state_visit_log(
    path: str | Path,
    state_visits: np.ndarray,
    *,
    num_agents: int,
    num_prices: int,
    memory: int,
) -> Path:
    """Write one count for every remembered joint-action state.

    State columns are ordered newest to oldest. With two agents and memory two,
    the file has ``15**4`` rows describing ``(a1_t-1, a2_t-1, a1_t-2, a2_t-2)``.
    """
    if num_agents != 2 or num_prices < 2 or memory < 0:
        raise ValueError("State visit logging currently supports two agents and non-negative memory")
    counts = np.asarray(state_visits, dtype=np.int64)
    state_width = num_agents * memory
    expected_states = num_prices**state_width
    if counts.size != expected_states:
        raise ValueError(f"State visit log has {counts.size} entries; expected {expected_states}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        f"agent_{agent + 1}_price_t_minus_{lag}_index"
        for lag in range(1, memory + 1)
        for agent in range(num_agents)
    ]
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([*columns, "state_visits"])
        for state_index, count in enumerate(counts):
            remaining = state_index
            state = [0] * state_width
            for position in range(state_width - 1, -1, -1):
                state[position] = remaining % num_prices
                remaining //= num_prices
            writer.writerow([*(value + 1 for value in state), int(count)])
    return target
