"""Common data contracts for policies used in taxi-market experiments.

The market environment deliberately stays independent from learning methods:
it accepts surge multipliers and returns the economic outcome.  These small
types give every policy the same view of that outcome, so a runner can train
Q-learning, EXP3, EXP4, or PPO without containing algorithm-specific updates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class MarketObservation:
    """Information available to one platform immediately before it prices.

    ``inventory_states`` is currently the only feature consumed by the
    tabular Q-learner. Day and hour are included now so contextual policies can
    use the same interface without changing the runner later.
    """

    inventory_states: np.ndarray
    day: int
    hour: int


@dataclass(frozen=True)
class PolicyAction:
    """Discrete price-action indexes selected by one platform across zones."""

    indexes: np.ndarray


@dataclass(frozen=True)
class Transition:
    """One platform's outcome from one market step.

    The runner supplies raw zone revenue. Policies retain responsibility for
    any reward normalisation required by their own learning rule.
    """

    observation: MarketObservation
    action: PolicyAction
    zone_rewards: np.ndarray
    next_observation: MarketObservation
    done: bool
    info: Mapping[str, np.ndarray | int]


class PlatformPolicy(Protocol):
    """Minimal lifecycle implemented by every platform pricing policy."""

    def act(self, observation: MarketObservation) -> PolicyAction:
        """Choose one discrete price action for each pickup zone."""

    def observe(self, transition: Transition) -> None:
        """Receive a transition after the market has advanced one step."""

    def end_episode(self) -> None:
        """Finish episode-level work, such as a PPO batch update."""


class FreezablePlatformPolicy(PlatformPolicy, Protocol):
    """A policy that can create an independent, non-learning snapshot."""

    def snapshot(self) -> PlatformPolicy:
        """Return a policy with fixed decision parameters and no learning."""
