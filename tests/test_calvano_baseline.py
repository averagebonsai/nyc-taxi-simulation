"""Small deterministic checks for the Calvano baseline port."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).parents[1] / "calvano-sourcecode"
sys.path.insert(0, str(PACKAGE))

from config import BatchConfig, ExperimentConfig, read_input
from model import BaselineGame
from reporting import (
    write_eqm_input,
    write_eqm_plot,
    write_figure4_input,
    write_impulse_response_csv,
    write_impulse_response_plot,
    write_impulse_response_plots,
    write_state_visit_log,
    write_state_action_visit_logs,
    write_training_visit_log,
)
from simulation import SessionResult, impulse_response, run_experiment


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
    state_action_visits = write_state_action_visit_logs(
        tmp_path / "A_trainingStateActionVisits",
        [session.state_action_visits for session in sessions],
        num_agents=2,
        num_prices=5,
        memory=1,
    )
    equilibrium_data = write_eqm_input(tmp_path, summary, impulse_cycles=2)
    equilibrium_plot = write_eqm_plot(equilibrium_data)
    impulse_data = write_impulse_response_csv(tmp_path / "impulse.csv", summary, 1.47293, 1.92498)
    impulse_plot = write_impulse_response_plot(impulse_data, "Impulse response", tmp_path / "impulse.pdf")
    impulse_pdf, impulse_png = write_impulse_response_plots(impulse_data, "Impulse response", tmp_path / "impulse_both")
    header, values = output.read_text(encoding="utf-8").splitlines()
    assert len(sessions) == 3
    assert "AggrDevPriceShockPer030" in header
    assert "AggrNonDevPriceShockPer030" in header
    assert len(header.split()) == len(values.split()) == 61
    assert sum(result.joint_action_visits.sum() for result in sessions) == sum(result.iterations for result in sessions)
    assert int(np.asarray(summary["JointActionVisits"]).sum()) == sum(result.iterations for result in sessions)
    assert len(visits.read_text(encoding="utf-8").splitlines()) == 26
    assert len(state_visits.read_text(encoding="utf-8").splitlines()) == 26
    assert len(state_action_visits) == 3
    state_action_rows = state_action_visits[0].read_text(encoding="utf-8").splitlines()
    assert state_action_rows[0] == "agent_num,agent_1_price_index,agent_2_price_index,chosen_action,count"
    assert len(state_action_rows) == 2 * 5 * 5 * 5 + 1
    assert sum(int(row.rsplit(",", 1)[1]) for row in state_action_rows[1:]) == 2 * sessions[0].iterations
    assert equilibrium_data.name == "2_eqm.csv"
    assert equilibrium_plot.name == "2_eqm.png"
    assert equilibrium_plot.exists()
    assert impulse_plot.exists()
    assert impulse_pdf.exists()
    assert impulse_png.exists()


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

    archive = tmp_path / "q_tables"
    _, sessions, summary = run_experiment(
        batch, experiment, seed=3, impulse_periods=2, save_q_tables=archive
    )
    _, loaded_sessions, loaded_summary = run_experiment(
        batch, experiment, impulse_periods=2, load_q_tables=archive
    )
    (archive / "session_0000.npz").unlink()
    _, resumed_sessions, resumed_summary = run_experiment(
        batch, experiment, seed=3, impulse_periods=2, resume_q_tables=archive
    )
    output = write_state_visit_log(
        tmp_path / "A_trainingStateVisits.csv", summary["StateVisits"], num_agents=2, num_prices=3, memory=2
    )
    assert sum(result.state_visits.sum() for result in sessions) == sum(result.iterations for result in sessions)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3**4 + 1
    assert (archive / "manifest.json").exists()
    assert (archive / "session_0000.npz").exists()
    assert loaded_sessions[0].state == sessions[0].state
    assert np.array_equal(loaded_sessions[0].policy, sessions[0].policy)
    assert np.allclose(loaded_summary["AggrDevPriceShock"], summary["AggrDevPriceShock"])
    assert np.array_equal(resumed_sessions[0].policy, sessions[0].policy)
    assert np.allclose(resumed_summary["AggrDevPriceShock"], summary["AggrDevPriceShock"])


def test_three_period_static_best_response_precedes_policy_recovery() -> None:
    """Each impulse cycle can hold the static best response for three turns."""
    batch = BatchConfig(
        num_experiments=1, total_experiments=1, num_cores=1, num_sessions=1,
        iterations_per_episode=2, max_episodes=3, performance_period_episodes=1,
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
    game = BaselineGame(batch, experiment)
    policy = np.full((game.num_states, batch.num_agents), 4, dtype=int)
    result = SessionResult(
        policy=policy,
        state=(4, 4),
        converged=True,
        iterations=1,
        joint_action_visits=np.zeros((5, 5), dtype=np.int64),
        state_visits=np.zeros(game.num_states, dtype=np.int64),
        state_action_visits=np.zeros((2, game.num_states, 5), dtype=np.int64),
    )

    deviator_prices, rival_prices, _ = impulse_response(game, result, periods=15, deviation_periods=3)
    expected_best_response = max(
        range(batch.num_prices),
        key=lambda price: game.profits[game.action_index(np.array([price, 4])), 0],
    )
    assert np.allclose(deviator_prices[:3], game.grids[expected_best_response, 0])
    assert np.allclose(rival_prices[:3], game.grids[4, 1])
    assert np.isclose(deviator_prices[3], game.grids[4, 0])


def test_zero_period_input_has_one_stateless_state_and_visit_log(tmp_path: Path) -> None:
    batch, experiments = read_input(PACKAGE / "A_InputParametersZeroPeriod.txt")
    game = BaselineGame(batch, experiments[0])
    output = write_state_visit_log(
        tmp_path / "A_trainingStateVisits.csv", np.array([12]), num_agents=2, num_prices=10, memory=0
    )
    assert game.num_states == 1
    assert game.state_index(()) == 0
    assert game.next_state((), np.array([2, 3])) == ()
    assert output.read_text(encoding="utf-8").splitlines() == ["state_visits", "12"]


def test_opponent_only_state_uses_the_rival_previous_action(tmp_path: Path) -> None:
    batch, experiments = read_input(PACKAGE / "A_InputParametersOpponentOnly.txt")
    batch = replace(
        batch,
        state_representation="opponent",
        num_sessions=1,
        iterations_per_episode=2,
        max_episodes=3,
    )
    game = BaselineGame(batch, experiments[0])
    state = (2, 7)
    assert game.num_states == 10
    assert game.state_index(state, agent=0) == 7
    assert game.state_index(state, agent=1) == 2

    _, sessions, summary = run_experiment(batch, experiments[0], seed=9, impulse_periods=2)
    output = write_state_visit_log(
        tmp_path / "A_trainingStateVisits.csv",
        summary["StateVisits"],
        num_agents=2,
        num_prices=10,
        memory=1,
        state_representation="opponent",
    )
    assert int(np.asarray(summary["StateVisits"]).sum()) == 2 * sum(result.iterations for result in sessions)
    assert output.read_text(encoding="utf-8").splitlines()[0] == "opponent_previous_price_index,state_visits"
