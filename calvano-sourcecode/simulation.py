"""Q-learning and impulse-response routines for the baseline model."""

from __future__ import annotations

# A ProcessPool worker represents one independent training session.  Limiting
# numerical-library threads prevents each worker from oversubscribing a cloud
# VM's CPUs (for example, 50 workers each attempting to use every core).
import os

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

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
    state_action_visits: np.ndarray


@dataclass(frozen=True)
class SessionFailure:
    """A recoverable failed seed; completed sessions remain checkpointed."""

    session: int
    seed: int
    diagnostic_path: Path


_ARCHIVE_MANIFEST = "manifest.json"


def _archive_metadata(game: BaselineGame, session_count: int, base_seed: int) -> dict[str, object]:
    """Return the compatibility information stored with a Q-table archive."""
    return {
        "schema_version": 3,
        "session_count": session_count,
        "base_seed": base_seed,
        "num_agents": game.batch.num_agents,
        "num_prices": game.batch.num_prices,
        "memory": game.batch.memory,
        "state_representation": game.batch.state_representation,
        "num_states": game.num_states,
        "price_grids": game.grids.tolist(),
        "training_parameters": {
            "iterations_per_episode": game.batch.iterations_per_episode,
            "max_episodes": game.batch.max_episodes,
            "performance_period_episodes": game.batch.performance_period_episodes,
            "exploration_type": game.batch.exploration_type,
            "alpha": game.experiment.alpha.tolist(),
            "exploration_m": game.experiment.exploration_m.tolist(),
            "discount": game.experiment.discount,
            "q_initialization": list(game.experiment.q_initialization),
            "q_initialization_parameters": game.experiment.q_initialization_parameters.tolist(),
            "payoffs": game.profits.tolist(),
        },
    }


def _prepare_q_archive(directory: str | Path, game: BaselineGame, session_count: int, base_seed: int) -> Path:
    """Create an empty archive directory and its compatibility manifest."""
    target = Path(directory)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Q-table archive directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    (target / _ARCHIVE_MANIFEST).write_text(
        json.dumps(_archive_metadata(game, session_count, base_seed), indent=2) + "\n", encoding="utf-8"
    )
    return target


def _session_archive_path(directory: Path, session: int) -> Path:
    return directory / f"session_{session:04d}.npz"


def _diagnostic_path(directory: Path, session: int) -> Path:
    return directory / f"session_{session:04d}.error.json"


def _save_session_q_table(path: Path, q: np.ndarray, result: SessionResult, seed: int) -> None:
    """Persist one trained session without returning its large Q-table to the parent."""
    temporary = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        q_values=q,
        policy=result.policy,
        state=np.asarray(result.state, dtype=np.int16),
        seed=np.asarray(seed, dtype=np.int64),
        converged=np.asarray(result.converged),
        iterations=np.asarray(result.iterations, dtype=np.int64),
        joint_action_visits=result.joint_action_visits,
        state_visits=result.state_visits,
        state_action_visits=result.state_action_visits,
    )
    os.replace(temporary, path)


def _write_failure_diagnostic(directory: Path, session: int, seed: int, error: BaseException) -> Path:
    """Write a per-seed traceback that survives a resumed run."""
    target = _diagnostic_path(directory, session)
    temporary = target.with_name(f"{target.stem}.tmp.json")
    payload = {
        "session": session,
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def _load_session_q_table(game: BaselineGame, path: Path, expected_seed: int | None = None) -> SessionResult:
    """Load one session's learned policy and validate the persisted Q-table."""
    expected_q_shape = (game.batch.num_agents, game.num_states, game.batch.num_prices)
    with np.load(path, allow_pickle=False) as archive:
        q = archive["q_values"]
        policy = archive["policy"]
        state = tuple(int(value) for value in archive["state"])
        seed = int(archive["seed"])
        converged = bool(archive["converged"])
        iterations = int(archive["iterations"])
        joint_action_visits = archive["joint_action_visits"]
        state_visits = archive["state_visits"]
        # Archives written before state-action logging retain compatibility for
        # post-training analysis, but do not contain the requested counts.
        state_action_visits = archive["state_action_visits"] if "state_action_visits" in archive.files else np.zeros(
            (game.batch.num_agents, game.num_states, game.batch.num_prices), dtype=np.int64
        )
    if q.shape != expected_q_shape or not np.isfinite(q).all():
        raise ValueError(f"{path} does not contain a valid Q-table for this configuration")
    if policy.shape != (game.num_states, game.batch.num_agents):
        raise ValueError(f"{path} does not contain a valid learned policy")
    if (
        len(state) != game.state_width
        or any(value < 0 or value >= game.batch.num_prices for value in state)
        or any(game.state_index(state, agent) >= game.num_states for agent in range(game.batch.num_agents))
    ):
        raise ValueError(f"{path} does not contain a valid terminal state")
    if joint_action_visits.shape != (game.batch.num_prices,) * game.batch.num_agents:
        raise ValueError(f"{path} does not contain valid joint-action visit counts")
    if state_visits.shape != (game.num_states,):
        raise ValueError(f"{path} does not contain valid state visit counts")
    if state_action_visits.shape != (game.batch.num_agents, game.num_states, game.batch.num_prices):
        raise ValueError(f"{path} does not contain valid state-action visit counts")
    if np.any(policy < 0) or np.any(policy >= game.batch.num_prices):
        raise ValueError(f"{path} contains policy actions outside the action grid")
    if seed < 0:
        raise ValueError(f"{path} contains an invalid seed")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError(f"{path} was trained with seed {seed}; expected {expected_seed}")
    return SessionResult(policy, state, converged, iterations, joint_action_visits, state_visits, state_action_visits)


def _read_archive_manifest(game: BaselineGame, directory: str | Path) -> tuple[Path, dict[str, object]]:
    """Read and validate an archive manifest against the current game."""
    source = Path(directory)
    try:
        metadata = json.loads((source / _ARCHIVE_MANIFEST).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"{source} is not a valid Q-table archive") from error
    session_count = int(metadata.get("session_count", 0))
    base_seed = int(metadata.get("base_seed", -1))
    # Archives written before alternative state representations existed are
    # joint-state archives.  Preserve their compatibility with this version.
    metadata.setdefault("state_representation", "joint")
    expected = _archive_metadata(game, session_count, base_seed)
    for key in ("schema_version", "num_agents", "num_prices", "memory", "state_representation", "num_states", "price_grids", "training_parameters"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"Q-table archive {source} is incompatible with the current game ({key} differs)")
    if session_count < 1:
        raise ValueError(f"Q-table archive {source} has no sessions")
    if base_seed < 0:
        raise ValueError(f"Q-table archive {source} has an invalid base seed")
    return source, metadata


def load_q_table_archive(game: BaselineGame, directory: str | Path) -> list[SessionResult]:
    """Load trained sessions for post-training simulation without Q-learning."""
    source, metadata = _read_archive_manifest(game, directory)
    return [_load_session_q_table(game, _session_archive_path(source, session)) for session in range(int(metadata["session_count"]))]


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
    # The usual O initialization has a unique best action, so select every
    # state's policy in one NumPy operation.  Retain random tie-breaking only
    # for the comparatively rare tied rows (or all rows for U initialization).
    policy = np.argmax(q, axis=2).T.astype(int, copy=False)
    for agent in range(batch.num_agents):
        values = q[agent]
        maxima = values.max(axis=1, keepdims=True)
        tied_states = np.flatnonzero(np.isclose(values, maxima, rtol=0.0, atol=1e-14).sum(axis=1) > 1)
        for state in tied_states:
            policy[state, agent] = _argmax_random(values[state], rng)
    return q, policy


def _choose_action(game: BaselineGame, policy: np.ndarray, q: np.ndarray, state_index: np.ndarray, epsilon: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    batch = game.batch
    action = np.empty(batch.num_agents, dtype=int)
    if batch.exploration_type == 1:
        for agent in range(batch.num_agents):
            action[agent] = rng.integers(batch.num_prices) if rng.random() <= epsilon[agent] else policy[state_index[agent], agent]
    else:
        for agent in range(batch.num_agents):
            temperature = max(epsilon[agent], np.finfo(float).tiny)
            values = q[agent, state_index[agent]]
            weights = np.exp((values - values.max()) / temperature)
            action[agent] = rng.choice(batch.num_prices, p=weights / weights.sum())
    return action


def run_session(game: BaselineGame, seed: int, q_archive_path: Path | None = None) -> SessionResult:
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
    state_action_visits = np.zeros((batch.num_agents, game.num_states, batch.num_prices), dtype=np.int64)
    agents = np.arange(batch.num_agents)
    for iteration in range(1, batch.max_iterations + 1):
        state_index = np.asarray([game.state_index(state, agent) for agent in agents])
        action = _choose_action(game, policy, q, state_index, epsilon, rng)
        action_index = game.action_index(action)
        if batch.state_representation == "opponent":
            for index in state_index:
                state_visits[index] += 1
        else:
            state_visits[state_index[0]] += 1
        for agent in agents:
            state_action_visits[agent, state_index[agent], action[agent]] += 1
        joint_action_visits[tuple(action)] += 1
        next_state = game.next_state(state, action)
        next_state_index = np.asarray([game.state_index(next_state, agent) for agent in agents])
        old_actions = policy[state_index, agents].copy()
        old_values = q[agents, state_index, action]
        # Pair each agent with its own next-state row.  For the usual joint
        # state these indices happen to match; for opponent-only state they do
        # not, so ``q[:, next_state_index]`` would form a Cartesian product.
        next_max = q[agents, next_state_index, :].max(axis=1)
        targets = game.profits[action_index] + experiment.discount * next_max
        q[agents, state_index, action] = old_values + experiment.alpha * (targets - old_values)
        for agent in range(batch.num_agents):
            policy[state_index[agent], agent] = _argmax_random(q[agent, state_index[agent]], rng)
        unchanged = bool(np.array_equal(old_actions, policy[state_index, agents]))
        stable = stable + 1 if unchanged else 1
        if stable >= batch.stability_iterations:
            result = SessionResult(policy, state, True, iteration, joint_action_visits, state_visits, state_action_visits)
            if q_archive_path is not None:
                _save_session_q_table(q_archive_path, q, result, seed)
            return result
        state = next_state
        epsilon *= decay
    result = SessionResult(policy, state, False, batch.max_iterations, joint_action_visits, state_visits, state_action_visits)
    if q_archive_path is not None:
        _save_session_q_table(q_archive_path, q, result, seed)
    return result


def _cycle(game: BaselineGame, result: SessionResult) -> list[tuple[int, ...]]:
    state, seen, path = result.state, {}, []
    while state not in seen:
        seen[state] = len(path)
        path.append(state)
        state = game.next_state(state, game.policy_action(result.policy, state))
    return path[seen[state] :]


def impulse_response(
    game: BaselineGame,
    result: SessionResult,
    periods: int = 15,
    cycles: int = 1,
    deviation_grid_steps: int | None = None,
    deviation_periods: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute repeated unilateral static-best-response impulse-response paths.

    For each cycle, one fixed deviator is forced to its static best-response
    price for ``deviation_periods`` periods. Both agents then follow their
    learned policies for the remainder of the cycle before the same deviator is
    shocked again. ``deviation_periods=1`` reproduces the original Figure 4
    analysis.
    """
    if periods < 1:
        raise ValueError("periods must be at least 1")
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    if deviation_periods < 1 or deviation_periods > periods:
        raise ValueError("deviation_periods must be between 1 and periods")
    if deviation_grid_steps is not None and deviation_grid_steps < 1:
        raise ValueError("deviation_grid_steps must be positive")
    dev_prices: list[np.ndarray] = []
    non_dev_prices: list[np.ndarray] = []
    pre_prices: list[float] = []
    for deviator in range(game.batch.num_agents):
        for initial_state in _cycle(game, result):
            state = initial_state
            baseline_action = game.policy_action(result.policy, state)
            deviator_path, rival_path = [], []
            for _ in range(cycles):
                for period in range(periods):
                    learned_action = game.policy_action(result.policy, state)
                    action = learned_action.copy()
                    if period < deviation_periods:
                        if deviation_grid_steps is None:
                            candidate_profit = []
                            for price in range(game.batch.num_prices):
                                candidate = learned_action.copy()
                                candidate[deviator] = price
                                candidate_profit.append(game.profits[game.action_index(candidate), deviator])
                            action[deviator] = int(np.argmax(candidate_profit))
                        else:
                            action[deviator] = max(0, action[deviator] - deviation_grid_steps)
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
    save_q_tables: str | Path | None = None,
    load_q_tables: str | Path | None = None,
    resume_q_tables: str | Path | None = None,
    deviation_grid_steps: int | None = None,
    deviation_periods: int = 1,
) -> tuple[BaselineGame, list[SessionResult], dict[str, np.ndarray | float]]:
    """Train or reload all sessions, then aggregate the impulse-response series."""
    archive_modes = sum(value is not None for value in (save_q_tables, load_q_tables, resume_q_tables))
    if archive_modes > 1:
        raise ValueError("Choose only one of save_q_tables, load_q_tables, or resume_q_tables")
    game = BaselineGame(batch, experiment)
    if load_q_tables is not None:
        results = load_q_table_archive(game, load_q_tables)
    else:
        completed: dict[int, SessionResult] = {}
        if resume_q_tables is not None:
            archive_directory, metadata = _read_archive_manifest(game, resume_q_tables)
            if int(metadata["session_count"]) != batch.num_sessions:
                raise ValueError("The archive's session count differs from the current input file")
            if int(metadata["base_seed"]) != seed:
                raise ValueError("The archive's base seed differs from --seed; use the original seed to resume")
            for session in range(batch.num_sessions):
                checkpoint = _session_archive_path(archive_directory, session)
                if checkpoint.exists():
                    completed[session] = _load_session_q_table(game, checkpoint, seed + session)
        else:
            archive_directory = _prepare_q_archive(save_q_tables, game, batch.num_sessions, seed) if save_q_tables is not None else None

        pending_sessions = [session for session in range(batch.num_sessions) if session not in completed]
        workers = min(batch.num_cores, len(pending_sessions), os.cpu_count() or 1)
        session_arguments = [
            (game, session, seed + session, _session_archive_path(archive_directory, session) if archive_directory is not None else None)
            for session in pending_sessions
        ]
        if workers == 0:
            outcomes: list[SessionResult | SessionFailure] = []
        elif workers == 1:
            outcomes = [_run_session_worker(arguments) for arguments in session_arguments]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                outcomes = list(executor.map(_run_session_worker, session_arguments))
        failures = [outcome for outcome in outcomes if isinstance(outcome, SessionFailure)]
        if failures:
            diagnostics = ", ".join(str(failure.diagnostic_path) for failure in failures)
            raise RuntimeError(
                f"{len(failures)} session(s) failed; completed sessions are checkpointed. "
                f"Inspect {diagnostics} and rerun with --resume-q-tables."
            )
        for session, outcome in zip(pending_sessions, outcomes):
            completed[session] = outcome
        results = [completed[session] for session in range(batch.num_sessions)]
    eligible = [result for result in results if result.converged] or results
    responses = [
        impulse_response(
            game,
            result,
            impulse_periods,
            impulse_cycles,
            deviation_grid_steps,
            deviation_periods,
        )
        for result in eligible
    ]
    return game, results, {
        "AggrPricePre": float(np.mean([response[2] for response in responses])),
        "AggrDevPriceShock": np.mean([response[0] for response in responses], axis=0),
        "AggrNonDevPriceShock": np.mean([response[1] for response in responses], axis=0),
        "JointActionVisits": np.sum([result.joint_action_visits for result in results], axis=0),
        "StateVisits": np.sum([result.state_visits for result in results], axis=0),
        "StateActionVisits": np.sum([result.state_action_visits for result in results], axis=0),
    }


def _run_session_worker(
    arguments: tuple[BaselineGame, int, int, Path | None],
) -> SessionResult | SessionFailure:
    """Run one seed and retain a diagnostic instead of discarding other sessions."""
    game, session, seed, q_archive_path = arguments
    try:
        return run_session(game, seed, q_archive_path)
    except Exception as error:
        if q_archive_path is None:
            raise
        return SessionFailure(session, seed, _write_failure_diagnostic(q_archive_path.parent, session, seed, error))
