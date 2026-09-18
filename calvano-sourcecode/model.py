"""Payoff matrices and state/action encodings for the baseline model."""

from __future__ import annotations

from itertools import product

import numpy as np

try:
    from .config import BatchConfig, ExperimentConfig
except ImportError:  # Allows the CLI to be run as a script.
    from config import BatchConfig, ExperimentConfig


class BaselineGame:
    """Finite-action repeated pricing game used by the original baseline code."""

    def __init__(self, batch: BatchConfig, experiment: ExperimentConfig) -> None:
        self.batch, self.experiment = batch, experiment
        self.actions = np.asarray(list(product(range(batch.num_prices), repeat=batch.num_agents)), dtype=int)
        self.action_to_index = {tuple(action): index for index, action in enumerate(self.actions)}
        self.state_width = batch.num_agents * batch.memory
        self.num_states = batch.num_prices**self.state_width if batch.memory else 1
        self.grids = self._price_grids()
        self.profits = self._payoff_matrix()

    def _price_grids(self) -> np.ndarray:
        parameters = self.experiment.demand_parameters
        if self.batch.payoff_type == 1:
            lower = np.maximum(0.0, self.experiment.nash_prices - parameters[1] * (self.experiment.cooperative_prices - self.experiment.nash_prices))
            upper = np.maximum(0.0, self.experiment.cooperative_prices + parameters[2] * (self.experiment.cooperative_prices - self.experiment.nash_prices))
        else:
            costs = parameters[1 + self.batch.num_agents : 1 + 2 * self.batch.num_agents]
            extension = parameters[-2:]
            if np.all(extension > 0):
                lower = self.experiment.nash_prices - extension[0] * (self.experiment.cooperative_prices - self.experiment.nash_prices)
                upper = self.experiment.cooperative_prices + extension[1] * (self.experiment.cooperative_prices - self.experiment.nash_prices)
            else:
                lower = costs + extension[0] * costs
                upper = self.experiment.cooperative_prices + extension[1] * (self.experiment.cooperative_prices - self.experiment.nash_prices)
        return np.column_stack([np.linspace(lower[agent], upper[agent], self.batch.num_prices) for agent in range(self.batch.num_agents)])

    def _payoff_matrix(self) -> np.ndarray:
        prices = np.vstack([self.grids[action, np.arange(self.batch.num_agents)] for action in self.actions])
        if self.batch.payoff_type == 1:
            gamma = self.experiment.demand_parameters[0]
            if self.batch.num_agents != 2:
                raise ValueError("Singh-Vives demand is defined only for two agents")
            demand = np.empty_like(prices)
            demand[:, 0] = np.clip((1 - gamma - prices[:, 0] + gamma * prices[:, 1]) / (1 - gamma**2), 0, 1 - prices[:, 0])
            demand[:, 1] = np.clip((1 - gamma - prices[:, 1] + gamma * prices[:, 0]) / (1 - gamma**2), 0, 1 - prices[:, 1])
            return prices * demand
        a0 = self.experiment.demand_parameters[0]
        values = self.experiment.demand_parameters[1 : 1 + self.batch.num_agents]
        costs = self.experiment.demand_parameters[1 + self.batch.num_agents : 1 + 2 * self.batch.num_agents]
        mu = self.experiment.demand_parameters[1 + 2 * self.batch.num_agents]
        utilities = np.exp((values - prices) / mu)
        demand = utilities / (utilities.sum(axis=1, keepdims=True) + np.exp(a0 / mu))
        return (prices - costs) * demand

    def action_index(self, action: np.ndarray) -> int:
        return self.action_to_index[tuple(int(value) for value in action)]

    def state_index(self, state: tuple[int, ...]) -> int:
        index = 0
        for value in state:
            index = index * self.batch.num_prices + value
        return index

    def next_state(self, state: tuple[int, ...], action: np.ndarray) -> tuple[int, ...]:
        if not self.batch.memory:
            return ()
        return tuple(int(value) for value in action) + state[: self.state_width - self.batch.num_agents]
