"""Command-line interface shared by Courthoud freeze/unfreeze variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .experiments.config import LearningConfig
from .experiments.courthoud import CourthoudConfig, N_FIRMS, run_courthoud_experiment
from .experiments.courthoud_reporting import load_monopoly_benchmark, save_courthoud_outputs
from .market.assets import load_simulation_assets
from .market.config import SimulationConfig, parse_fleet_sizes


def parse_args(default_reset_frozen: bool) -> argparse.Namespace:
    """Retain the legacy flags while mapping them to modular configuration."""
    parser = argparse.ArgumentParser(description="Run a modular Courthoud freeze/unfreeze defection experiment.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/courthoud_freeze_unfreeze_monopoly_reset" if default_reset_frozen else "experiments/courthoud_freeze_unfreeze_monopoly"))
    parser.add_argument("--episodes", type=int, default=100, help="Baseline oligopoly training episodes.")
    parser.add_argument("--baseline-eval-episodes", type=int, default=70)
    parser.add_argument("--freeze-episodes", type=int, default=70)
    parser.add_argument("--unfreeze-at", type=int, default=None, help="Absolute timeline episode at which platforms unfreeze.")
    parser.add_argument("--post-unfreeze-episodes", type=int, default=70)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--taxis", type=int, nargs="+", default=[5000])
    parser.add_argument("--n-states", type=int, default=10)
    parser.add_argument("--bin-size", type=int, default=10)
    parser.add_argument("--theta", type=float, default=0.4)
    parser.add_argument("--start-day", type=int, default=0)
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.1, help="EXP3 exploration probability.")
    parser.add_argument("--eta", type=float, default=0.1, help="EXP3 learning rate.")
    parser.add_argument("--reward-normalizer", type=float, default=50_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-deviator", action="store_true")
    parser.add_argument("--deviator", type=int, default=0)
    parser.add_argument("--record-baseline-training", action="store_true", help="Include the initial all-firm learning phase in the plotted timeline.")
    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument("--reset-frozen", dest="reset_frozen", action="store_true")
    reset_group.add_argument("--keep-frozen-memory", dest="reset_frozen", action="store_false")
    parser.set_defaults(reset_frozen=default_reset_frozen)
    parser.add_argument("--ma-window", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--monopoly-csv", type=Path, default=None)
    parser.add_argument("--monopoly-column", type=str, default=None)
    parser.add_argument("--monopoly-scale", type=float, default=0.25)
    return parser.parse_args()


def main(default_reset_frozen: bool = False) -> None:
    args = parse_args(default_reset_frozen)
    baseline_offset = args.episodes if args.record_baseline_training else 0
    freeze_episodes = args.freeze_episodes if args.unfreeze_at is None else args.unfreeze_at - args.baseline_eval_episodes - baseline_offset
    if freeze_episodes < 1:
        raise ValueError("--unfreeze-at must be after the baseline evaluation phase.")
    deviator = int(np.random.default_rng(args.seed).integers(N_FIRMS)) if args.random_deviator else args.deviator
    simulation = SimulationConfig(
        fleet_sizes=parse_fleet_sizes(args.taxis),
        steps_per_episode=args.steps,
        n_inventory_states=args.n_states,
        inventory_bin_size=args.bin_size,
        price_sensitivity=args.theta,
        start_day=args.start_day,
        start_hour=args.start_hour,
    )
    learning = LearningConfig(
        episodes=args.episodes,
        exp3_exploration=args.gamma,
        exp3_learning_rate=args.eta,
        reward_normalizer=args.reward_normalizer,
        seed=args.seed,
    )
    courthoud = CourthoudConfig(
        baseline_training_episodes=args.episodes,
        evaluation_episodes=args.baseline_eval_episodes,
        freeze_episodes=freeze_episodes,
        post_unfreeze_episodes=args.post_unfreeze_episodes,
        deviator=deviator,
        reset_frozen_agents=args.reset_frozen,
        record_baseline_training=args.record_baseline_training,
        log_every=args.log_every,
    )
    assets = load_simulation_assets(args.data_dir)
    frozen = [firm + 1 for firm in range(N_FIRMS) if firm != deviator]
    print(f"deviator = Platform {deviator + 1}; frozen = {frozen}")
    result = run_courthoud_experiment(assets, simulation, learning, courthoud, progress=print)
    print("frozen-policy expected avg multipliers:", np.round(result.frozen_multipliers.mean(axis=1), 3))
    target_length = max(int(row["iteration"]) for row in result.records) + 1
    benchmark = load_monopoly_benchmark(args.monopoly_csv, target_length, args.monopoly_scale, args.monopoly_column)
    paths = save_courthoud_outputs(result, courthoud, args.out_dir, benchmark, args.ma_window, args.dpi)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
