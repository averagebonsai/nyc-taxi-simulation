"""Modular dynamic-pricing research pipeline.

Run the supported experiment entry point with:
    python -m src2.run_experiments --help
"""

from .experiments.config import LearningConfig
from .market.config import PRICE_ACTIONS, SimulationConfig

__all__ = ["PRICE_ACTIONS", "LearningConfig", "SimulationConfig"]
