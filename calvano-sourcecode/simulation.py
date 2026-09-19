"""Q-learning and impulse-response routines for the baseline model."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os

import numpy as np

try:
    from .config import BatchConfig, ExperimentConfig
    from .model import BaselineGame
except ImportError:  # Allows the CLI to be run as a script.
    from config import BatchConfig, ExperimentConfig
    from model import BaselineGame


@dataclass
class SessionResult:
    policy: np.ndarray
    state: tuple[int, ...]
    converged: bool
    iterations: int
    joint_action_visits: np.ndarray
    state_visits: np.ndarray


def _argmax_random(values: np.ndarray, rng: np.random.Generator) -> int:
    choices = np.flatnonzero(np.isclose(values, values.max(), rtol=0.0, atol=1e-14))
    return int(rng.choice(choices))


def _initial_q(game: BaselineGame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    batch, experiment = game.batch, game.experiment
    q = np.zeros((batch.num_agents, game.num_states, batch.num_prices), dtype=float)
    for agent, kind in enumerate(experiment.q_initialization):
        if kind == "O":
            for price in range(batch.num_prices):
                mask = game.actions[:, agent] == price
                q[agent, :, price] = game.profits[mask, agent].mean() / (1.0 - experiment.discount)
        elif kind == "R":
            low, high = experiment.q_initialization_parameters[agent]
            q[agent] = rng.uniform(low, high, size=(game.num_states, batch.num_prices))
        elif kind == "U":
            q[agent].fill(experiment.q_initialization_parameters[agent, 0])
        else:
            raise ValueError(f"Q initialization {kind!r} is not implemented; use O, R, or U")
    policy = np.empty((game.num_states, batch.num_agents), dtype=int)
    for state in range(game.num_states):
        for agent in range(batch.num_agents):
            policy[state, agent] = _argmax_random(q[agent, state], rng)
    return q, policy


def _choose_action(game: BaselineGame, policy: np.ndarray, q: np.ndarray, state_index: int, epsilon: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    batch = game.batch
    action = np.empty(batch.num_agents, dtype=int)
    if batch.exploration_type == 1:
        for agent in range(batch.num_agents):
            action[agent] = rng.integers(batch.num_prices) if rng.random() <= epsilon[agent] else policy[state_index, agent]
    else:
        for agent in range(batch.num_agents):
            temperature = max(epsilon[agent], np.finfo(float).tiny)
            values = q[agent, state_index]
            weights = np.exp((values - values.max()) / temperature)
            action[agent] = rng.choice(batch.num_prices, p=weights / weights.sum())
    return action


def run_session(game: BaselineGame, seed: int) -> SessionResult:
    """Run one learning replication using the original asynchronous Q update rule."""
    batch, experiment = game.batch, game.experiment
    rng = np.random.default_rng(seed)
    q, policy = _initial_q(game, rng)
    state = tuple(rng.integers(batch.num_prices, size=game.state_width).tolist())
    epsilon = np.ones(batch.num_agents) if batch.exploration_type == 1 else np.full(batch.num_agents, 1_000.0)
    decay = np.exp(-experiment.exploration_m / batch.iterations_per_episode) if batch.exploration_type == 1 else 1.0 - 0.1**experiment.exploration_m
    stable = 0
    joint_action_visits = np.zeros((batch.num_prices,) * batch.num_agents, dtype=np.int64)
    state_visits = np.zeros(game.num_states, dtype=np.int64)
    for iteration in range(1, batch.max_iterations + 1):
        state_index = game.state_index(state)
        action = _choose_action(game, policy, q, state_index, epsilon, rng)
        action_index = game.action_index(action)
        state_visits[state_index] += 1
        joint_action_visits[tuple(action)] += 1
        next_state = game.next_state(state, action)
        next_state_index = game.state_index(next_state)
        unchanged = True
        for agent in range(batch.num_agents):
            old_action = policy[state_index, agent]
            old_value = q[agent, state_index, action[agent]]
            target = game.profits[action_index, agent] + experiment.discount * q[agent, next_state_index].max()
            q[agent, state_index, action[agent]] = old_value + experiment.alpha[agent] * (target - old_value)
            policy[state_index, agent] = _argmax_random(q[agent, state_index], rng)
            unchanged = unchanged and old_action == policy[state_index, agent]
        stable = stable + 1 if unchanged else 1
        if stable >= batch.stability_iterations:
            return SessionResult(policy, state, True, iteration, joint_action_visits, state_visits)
        state = next_state
        epsilon *= decay
    return SessionResult(policy, state, False, batch.max_iterations, joint_action_visits, state_visits)


def _cycle(game: BaselineGame, result: SessionResult) -> list[tuple[int, ...]]:
    state, seen, path = result.state, {}, []
    while state not in seen:
        seen[state] = len(path)
        path.append(state)
        state = game.next_state(state, result.policy[game.state_index(state)])
    return path[seen[state] :]


def impulse_response(
    game: BaselineGame,
    result: SessionResult,
    periods: int = 15,
    cycles: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute repeated unilateral static-best-response impulse-response paths.

    For each cycle, one fixed deviator is forced to its static best-response
    price in the first period. Both agents then follow their learned policies for
    the remaining ``periods - 1`` periods before the same deviator is shocked
    again. ``cycles=1`` reproduces the original one-deviation analysis.
    """
    if periods < 1:
        raise ValueError("periods must be at least 1")
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    dev_prices: list[np.ndarray] = []
    non_dev_prices: list[np.ndarray] = []
    pre_prices: list[float] = []
    for deviator in range(game.batch.num_agents):
        for initial_state in _cycle(game, result):
            state = initial_state
            baseline_action = result.policy[game.state_index(state)].copy()
            deviator_path, rival_path = [], []
            for _ in range(cycles):
                learned_action = result.policy[game.state_index(state)].copy()
                candidate_profit = []
                for price in range(game.batch.num_prices):
                    candidate = learned_action.copy()
                    candidate[deviator] = price
                    candidate_profit.append(game.profits[game.action_index(candidate), deviator])
                action = learned_action.copy()
                action[deviator] = int(np.argmax(candidate_profit))
                for period in range(periods):
                    if period:
                        action = result.policy[game.state_index(state)].copy()
                    prices = game.grids[action, np.arange(game.batch.num_agents)]
                    deviator_path.append(prices[deviator])
                    rival_path.append(np.delete(prices, deviator).mean())
                    state = game.next_state(state, action)
            pre_prices.append(game.grids[baseline_action, np.arange(game.batch.num_agents)].mean())
            dev_prices.append(np.asarray(deviator_path))
            non_dev_prices.append(np.asarray(rival_path))
    return np.mean(dev_prices, axis=0), np.mean(non_dev_prices, axis=0), float(np.mean(pre_prices))


def run_experiment(
    batch: BatchConfig,
    experiment: ExperimentConfig,
    *,
    seed: int = 1,
    impulse_periods: int = 15,
    impulse_cycles: int = 1,
) -> tuple[BaselineGame, list[SessionResult], dict[str, np.ndarray | float]]:
    """Run all sessions and aggregate the Figure 4 impulse-response series."""
    game = BaselineGame(batch, experiment)
    workers = min(batch.num_cores, batch.num_sessions, os.cpu_count() or 1)
    session_seeds = [seed + session for session in range(batch.num_sessions)]
    if workers == 1:
        results = [run_session(game, session_seed) for session_seed in session_seeds]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_run_session_worker, [(game, session_seed) for session_seed in session_seeds]))
    eligible = [result for result in results if result.converged] or results
    responses = [impulse_response(game, result, impulse_periods, impulse_cycles) for result in eligible]
    return game, results, {
        "AggrPricePre": float(np.mean([response[2] for response in responses])),
        "AggrDevPriceShock": np.mean([response[0] for response in responses], axis=0),
        "AggrNonDevPriceShock": np.mean([response[1] for response in responses], axis=0),
        "JointActionVisits": np.sum([result.joint_action_visits for result in results], axis=0),
        "StateVisits": np.sum([result.state_visits for result in results], axis=0),
    }


def _run_session_worker(arguments: tuple[BaselineGame, int]) -> SessionResult:
    """Process-pool entry point; defined at module level so it is picklable."""
    game, seed = arguments
    return run_session(game, seed)
