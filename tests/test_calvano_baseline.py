"""Small deterministic checks for the Calvano baseline port."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).parents[1] / "calvano-sourcecode"
sys.path.insert(0, str(PACKAGE))

from config import BatchConfig, ExperimentConfig
from model import BaselineGame
from reporting import write_eqm_input, write_eqm_plot, write_figure4_input, write_state_visit_log, write_training_visit_log
from simulation import run_experiment


def test_small_baseline_run_writes_figure4_columns(tmp_path: Path) -> None:
    batch = BatchConfig(
        num_experiments=1, total_experiments=1, num_cores=2, num_sessions=3,
        iterations_per_episode=5, max_episodes=8, performance_period_episodes=1,
        num_agents=2, memory=1, num_prices=5, exploration_type=1, payoff_type=2,
        impulse_response_to_br=True, impulse_response_to_nash=0, impulse_response_to_all=False,
        equilibrium_check=False, q_gap_to_maximum=False, learning_trajectory=(0, 0), detailed_analysis=False,
    )
    experiment = ExperimentConfig(
        identifier=1, print_q=False, alpha=np.array([0.15, 0.15]), exploration_m=np.array([0.4, 0.4]),
        discount=0.95, demand_parameters=np.array([0.0, 2.0, 2.0, 1.0, 1.0, 0.25, 0.1, 0.1]),
        nash_prices=np.array([1.47293, 1.47293]), cooperative_prices=np.array([1.92498, 1.92498]),
        q_initialization=("O", "O"), q_initialization_parameters=np.zeros((2, 2)),
    )
    _, sessions, summary = run_experiment(batch, experiment, seed=7, impulse_periods=15, impulse_cycles=2)
    output = write_figure4_input(tmp_path / "A_irToBR.txt", summary)
    visits = write_training_visit_log(
        tmp_path / "A_trainingVisits.csv", summary["JointActionVisits"], summary["StateVisits"]
    )
    state_visits = write_state_visit_log(
        tmp_path / "A_trainingStateVisits.csv", summary["StateVisits"], num_agents=2, num_prices=5, memory=1
    )
    equilibrium_data = write_eqm_input(tmp_path, summary, impulse_cycles=2)
    equilibrium_plot = write_eqm_plot(equilibrium_data)
    header, values = output.read_text(encoding="utf-8").splitlines()
    assert len(sessions) == 3
    assert "AggrDevPriceShockPer030" in header
    assert "AggrNonDevPriceShockPer030" in header
    assert len(header.split()) == len(values.split()) == 61
    assert sum(result.joint_action_visits.sum() for result in sessions) == sum(result.iterations for result in sessions)
    assert int(np.asarray(summary["JointActionVisits"]).sum()) == sum(result.iterations for result in sessions)
    assert len(visits.read_text(encoding="utf-8").splitlines()) == 26
    assert len(state_visits.read_text(encoding="utf-8").splitlines()) == 26
    assert equilibrium_data.name == "2_eqm.csv"
    assert equilibrium_plot.name == "2_eqm.png"
    assert equilibrium_plot.exists()


def test_two_period_memory_encodes_and_logs_four_price_state(tmp_path: Path) -> None:
    batch = BatchConfig(
        num_experiments=1, total_experiments=1, num_cores=1, num_sessions=1,
        iterations_per_episode=2, max_episodes=3, performance_period_episodes=1,
        num_agents=2, memory=2, num_prices=3, exploration_type=1, payoff_type=2,
        impulse_response_to_br=True, impulse_response_to_nash=0, impulse_response_to_all=False,
        equilibrium_check=False, q_gap_to_maximum=False, learning_trajectory=(0, 0), detailed_analysis=False,
    )
    experiment = ExperimentConfig(
        identifier=1, print_q=False, alpha=np.array([0.15, 0.15]), exploration_m=np.array([0.4, 0.4]),
        discount=0.95, demand_parameters=np.array([0.0, 2.0, 2.0, 1.0, 1.0, 0.25, 0.1, 0.1]),
        nash_prices=np.array([1.47293, 1.47293]), cooperative_prices=np.array([1.92498, 1.92498]),
        q_initialization=("O", "O"), q_initialization_parameters=np.zeros((2, 2)),
    )
    game = BaselineGame(batch, experiment)
    state = (0, 1, 2, 0)
    assert game.num_states == 3**4
    assert game.state_from_index(game.state_index(state)) == state
    assert game.next_state(state, np.array([2, 1])) == (2, 1, 0, 1)

    _, sessions, summary = run_experiment(batch, experiment, seed=3, impulse_periods=2)
    output = write_state_visit_log(
        tmp_path / "A_trainingStateVisits.csv", summary["StateVisits"], num_agents=2, num_prices=3, memory=2
    )
    assert sum(result.state_visits.sum() for result in sessions) == sum(result.iterations for result in sessions)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3**4 + 1
