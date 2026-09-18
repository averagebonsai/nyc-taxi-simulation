"""Small deterministic checks for the Calvano baseline port."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).parents[1] / "calvano-sourcecode"
sys.path.insert(0, str(PACKAGE))

from config import BatchConfig, ExperimentConfig
from reporting import write_figure4_input
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
    _, sessions, summary = run_experiment(batch, experiment, seed=7, impulse_periods=15)
    output = write_figure4_input(tmp_path / "A_irToBR.txt", summary)
    header, values = output.read_text(encoding="utf-8").splitlines()
    assert len(sessions) == 3
    assert "AggrDevPriceShockPer015" in header
    assert "AggrNonDevPriceShockPer015" in header
    assert len(header.split()) == len(values.split()) == 31
