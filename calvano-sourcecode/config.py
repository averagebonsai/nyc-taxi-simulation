"""Parsing and validation for the baseline Fortran input format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BatchConfig:
    num_experiments: int
    total_experiments: int
    num_cores: int
    num_sessions: int
    iterations_per_episode: int
    max_episodes: int
    performance_period_episodes: float
    num_agents: int
    memory: int
    num_prices: int
    exploration_type: int
    payoff_type: int
    impulse_response_to_br: bool
    impulse_response_to_nash: int
    impulse_response_to_all: bool
    equilibrium_check: bool
    q_gap_to_maximum: bool
    learning_trajectory: tuple[int, int]
    detailed_analysis: bool

    @property
    def max_iterations(self) -> int:
        return self.iterations_per_episode * self.max_episodes

    @property
    def stability_iterations(self) -> int:
        return int(self.performance_period_episodes * self.iterations_per_episode)


@dataclass(frozen=True)
class ExperimentConfig:
    identifier: int
    print_q: bool
    alpha: np.ndarray
    exploration_m: np.ndarray
    discount: float
    demand_parameters: np.ndarray
    nash_prices: np.ndarray
    cooperative_prices: np.ndarray
    q_initialization: tuple[str, ...]
    q_initialization_parameters: np.ndarray


def _value_lines(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def _as_numbers(line: str, cast: type[int] | type[float]) -> list[int] | list[float]:
    return [cast(value) for value in line.split()]


def read_input(path: str | Path) -> tuple[BatchConfig, list[ExperimentConfig]]:
    """Read an unmodified baseline ``A_InputParameters.txt`` file."""
    source = Path(path)
    lines = _value_lines(source)
    if len(lines) < 38:
        raise ValueError(f"{source} is not a complete baseline input file")
    try:
        num_experiments, total_experiments = _as_numbers(lines[1], int)
        num_cores = int(lines[3])
        num_sessions = int(lines[5])
        iterations_per_episode = int(lines[7])
        max_episodes = int(lines[9])
        performance_period = float(lines[11])
        num_agents = int(lines[13])
        memory = int(lines[15])
        num_prices = int(lines[17])
        exploration_type = int(lines[19])
        payoff_type = int(lines[21])
        switches = [int(lines[index]) for index in (23, 25, 27, 29, 31)]
        trajectory = tuple(_as_numbers(lines[33], int))
        detailed_analysis = bool(int(lines[35]))
    except (IndexError, ValueError) as error:
        raise ValueError(f"Unable to parse baseline settings in {source}") from error
    if num_agents < 2 or num_prices < 2 or memory < 0 or num_cores < 1:
        raise ValueError("num_agents, num_prices, and num_cores must be positive; memory cannot be negative")
    if exploration_type not in (1, 2) or payoff_type not in (1, 2, 3) or len(trajectory) != 2:
        raise ValueError("unsupported exploration/payoff type or invalid trajectory setting")
    batch = BatchConfig(
        num_experiments, total_experiments, num_cores, num_sessions, iterations_per_episode, max_episodes,
        performance_period, num_agents, memory, num_prices, exploration_type, payoff_type,
        bool(switches[0]), switches[1], bool(switches[2]), bool(switches[3]), bool(switches[4]),
        (int(trajectory[0]), int(trajectory[1])), detailed_analysis,
    )
    demand_count = 3 if payoff_type == 1 else 2 * num_agents + 4
    experiments: list[ExperimentConfig] = []
    for row in lines[37:]:
        values = row.split()
        expected = 2 + 2 * num_agents + 1 + demand_count + 2 * num_agents + 3 * num_agents
        if len(values) != expected:
            raise ValueError(f"Experiment row has {len(values)} fields; expected {expected} for {num_agents} agents")
        cursor = 0
        identifier, print_q = int(values[cursor]), bool(int(values[cursor + 1]))
        cursor += 2
        alpha = np.asarray(values[cursor : cursor + num_agents], dtype=float)
        cursor += num_agents
        exploration_m = np.asarray(values[cursor : cursor + num_agents], dtype=float)
        cursor += num_agents
        discount = float(values[cursor])
        cursor += 1
        demand = np.asarray(values[cursor : cursor + demand_count], dtype=float)
        cursor += demand_count
        nash = np.asarray(values[cursor : cursor + num_agents], dtype=float)
        cursor += num_agents
        cooperative = np.asarray(values[cursor : cursor + num_agents], dtype=float)
        cursor += num_agents
        init_types: list[str] = []
        init_parameters = np.zeros((num_agents, num_agents), dtype=float)
        for agent in range(num_agents):
            init_types.append(values[cursor].upper())
            cursor += 1
            init_parameters[agent] = np.asarray(values[cursor : cursor + num_agents], dtype=float)
            cursor += num_agents
        experiments.append(ExperimentConfig(identifier, print_q, alpha, exploration_m, discount, demand, nash, cooperative, tuple(init_types), init_parameters))
    if len(experiments) < num_experiments:
        raise ValueError(f"Input requests {num_experiments} experiments but contains {len(experiments)} rows")
    return batch, experiments[:num_experiments]
