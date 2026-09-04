"""Focused checks for the supported modular simulation pipeline."""

from pathlib import Path

import numpy as np

from src2.experiments.config import LearningConfig
from src2.experiments.courthoud import CourthoudConfig, run_courthoud_experiment
from src2.experiments.courthoud_reporting import save_courthoud_outputs
from src2.experiments.runner import run_monopoly
from src2.market.assets import FareModel, SimulationAssets, load_simulation_assets
from src2.market.config import SimulationConfig
from src2.market.environment import TaxiMarketEnv
from src2.policies.algorithms import FrozenZoneExp3, ZoneExp3, ZoneQLearner
from src2.policies.base import MarketObservation


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def test_zone_qlearner_does_not_bootstrap_terminal_transition() -> None:
    """A terminal Q-learning target is reward only, never reward plus future value."""
    learner = ZoneQLearner(
        n_zones=2,
        n_states=2,
        n_actions=2,
        learning_rate=1.0,
        discount=0.9,
        epsilon=0.0,
        rng=np.random.default_rng(42),
    )
    learner.q_values[:, 1] = np.array([[3.0, 7.0], [11.0, 5.0]])
    states = np.array([0, 0])
    actions = np.array([0, 1])
    rewards = np.array([2.0, 4.0])
    next_states = np.array([1, 1])

    learner.update(states, actions, rewards, next_states, done=True)

    np.testing.assert_allclose(learner.q_values[np.arange(2), states, actions], rewards)


def test_zone_qlearner_bootstraps_non_terminal_transition() -> None:
    """A continuing transition retains the discounted maximum future value."""
    learner = ZoneQLearner(
        n_zones=1,
        n_states=2,
        n_actions=2,
        learning_rate=1.0,
        discount=0.9,
        epsilon=0.0,
        rng=np.random.default_rng(42),
    )
    learner.q_values[0, 1] = np.array([3.0, 7.0])

    learner.update(np.array([0]), np.array([0]), np.array([2.0]), np.array([1]))

    np.testing.assert_allclose(learner.q_values[0, 0, 0], 2.0 + 0.9 * 7.0)


def test_exp3_snapshot_samples_fixed_stochastic_policy() -> None:
    """A frozen EXP3 snapshot samples saved probabilities and cannot learn."""
    agent = ZoneExp3(
        n_zones=1,
        n_actions=2,
        exploration=0.0,
        learning_rate=1.0,
        reward_normalizer=10.0,
        rng=np.random.default_rng(42),
    )
    # These weights represent action probabilities of 0.2 and 0.8.
    agent.log_weights[0] = np.log(np.array([1.0, 4.0]))
    frozen = agent.snapshot()

    assert isinstance(frozen, FrozenZoneExp3)
    np.testing.assert_allclose(frozen.action_probabilities, np.array([[0.2, 0.8]]))

    # The live agent may keep learning, but its frozen copy must not change.
    agent.update(np.array([0]), np.array([10.0]))
    np.testing.assert_allclose(frozen.action_probabilities, np.array([[0.2, 0.8]]))

    observation = MarketObservation(inventory_states=np.array([0]), day=0, hour=0)
    sampled_actions = np.array([frozen.act(observation).indexes[0] for _ in range(1_000)])
    assert 0.74 < sampled_actions.mean() < 0.86


def test_prepared_assets_and_inventory_states_match_runtime_contract() -> None:
    """The committed prepared data exposes 262 zones and bounded inventory states."""
    assets = load_simulation_assets(DATA_DIR)
    config = SimulationConfig(fleet_sizes=(500,), n_inventory_states=10, inventory_bin_size=10)
    environment = TaxiMarketEnv(assets, config, n_firms=1, rng=np.random.default_rng(42))

    states = environment.reset()

    assert assets.n_zones == 262
    assert states.shape == (1, 262)
    assert np.all((0 <= states) & (states < config.n_inventory_states))


def test_monopoly_runner_smoke() -> None:
    """A minimal supported Q-learning experiment completes with expected shapes."""
    assets = load_simulation_assets(DATA_DIR)
    result = run_monopoly(
        assets,
        SimulationConfig(fleet_sizes=(500,), steps_per_episode=1, n_inventory_states=10, inventory_bin_size=10),
        LearningConfig(episodes=1, seed=42),
    )

    assert len(result.records) == 1
    assert result.multiplier_history.shape == (1, 1, 1, 262)
    assert np.isfinite(result.records[0]["revenue"])


class _UniformDestinationPredictor:
    """Minimal predictor used to keep Courthoud tests independent of prepared files."""

    def predict_proba_batch(self, pickup_zones: np.ndarray, day: int, hour: int) -> np.ndarray:
        return np.full((len(pickup_zones), len(pickup_zones)), 1.0 / len(pickup_zones))


def test_courthoud_freeze_unfreeze_records_and_reports(tmp_path: Path) -> None:
    """The reset variant runs every phase while freezing only non-deviators."""
    assets = SimulationAssets(
        zone_ids=np.array([1, 2]),
        demand=np.full((2, 7, 24), 2.0),
        fares=np.full((2, 2), 10.0),
        destination_to_zone=np.array([0, 1]),
        predictor=_UniformDestinationPredictor(),
        fare_model=FareModel(price_per_mile=1.0, intercept=5.0),
        _destination_cache={},
    )
    simulation = SimulationConfig(fleet_sizes=(40,), steps_per_episode=1, n_inventory_states=5, inventory_bin_size=2)
    learning = LearningConfig(episodes=1, seed=42, reward_normalizer=100.0)
    intervention = CourthoudConfig(
        baseline_training_episodes=1,
        evaluation_episodes=0,
        freeze_episodes=1,
        post_unfreeze_episodes=1,
        deviator=1,
        reset_frozen_agents=True,
        record_baseline_training=True,
    )

    result = run_courthoud_experiment(assets, simulation, learning, intervention)

    assert len(result.records) == 12
    assert result.frozen_multipliers.shape == (4, 2)
    frozen_rows = [row for row in result.records if row["iteration"] == 1]
    assert [row["is_frozen"] for row in frozen_rows] == [1, 0, 1, 1]
    assert all(np.isfinite(float(row["revenue"])) for row in result.records)

    outputs = save_courthoud_outputs(result, intervention, tmp_path, monopoly_benchmark=None, moving_average_window=1)
    assert all(path.exists() for path in outputs.values())
