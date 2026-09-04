"""Reusable, stateful learning agents for zone-level dynamic pricing."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .base import MarketObservation, PlatformPolicy, PolicyAction, Transition


@dataclass
class ZoneQLearner:
    """Independent tabular Q-learning, one state/action table per pickup zone.

    The zones are factorised deliberately: learning a joint policy over every
    zone's multiplier would require an infeasibly large action space.

    The value tensor has shape ``(n_zones, n_inventory_states, n_actions)``.
    Its dimensions are derived from the prepared assets and market config.
    """

    n_zones: int
    n_states: int
    n_actions: int
    learning_rate: float
    discount: float
    epsilon: float
    rng: np.random.Generator
    reward_normalizer: float = 1.0
    q_values: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.reward_normalizer <= 0:
            raise ValueError("reward_normalizer must be positive.")
        self.q_values = np.zeros((self.n_zones, self.n_states, self.n_actions), dtype=float)

    def select_actions(self, states: np.ndarray) -> np.ndarray:
        """Choose one price-action index per zone using epsilon-greedy policy."""
        states = np.asarray(states, dtype=int)
        greedy = np.argmax(self.q_values[np.arange(self.n_zones), states], axis=1)
        explore = self.rng.random(self.n_zones) < self.epsilon
        random_actions = self.rng.integers(self.n_actions, size=self.n_zones)
        return np.where(explore, random_actions, greedy).astype(int)

    def act(self, observation: MarketObservation) -> PolicyAction:
        """Adapt Q-learning's inventory-state action rule to the common API."""
        return PolicyAction(self.select_actions(observation.inventory_states))

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        done: bool = False,
    ) -> None:
        """Apply an independent Q-learning update for every zone.

        Terminal transitions have no future value, so their target is the
        immediate reward rather than a bootstrapped Q-value.
        """
        zones = np.arange(self.n_zones)
        rewards = np.asarray(rewards, dtype=float)
        future_values = self.q_values[zones, next_states].max(axis=1)
        targets = rewards if done else rewards + self.discount * future_values
        current = self.q_values[zones, states, actions]
        self.q_values[zones, states, actions] += self.learning_rate * (targets - current)

    def observe(self, transition: Transition) -> None:
        """Apply the Q update from the generic market transition.

        This preserves the existing runner behaviour: Q-values receive bounded
        rewards while experiment records retain dollar revenue.
        """
        self.update(
            transition.observation.inventory_states,
            transition.action.indexes,
            np.clip(transition.zone_rewards / self.reward_normalizer, 0.0, 1.0),
            transition.next_observation.inventory_states,
            done=transition.done,
        )

    def end_episode(self) -> None:
        """Q-learning updates step-by-step, so no episode-level work is needed."""


@dataclass
class ZoneExp3:
    """Numerically stable reward-form EXP3 with one bandit per pickup zone.

    Rewards must be raw non-negative revenue.  Scaling is owned here rather
    than scattered across runners, keeping experiments comparable.
    """

    n_zones: int
    n_actions: int
    exploration: float
    learning_rate: float
    reward_normalizer: float
    rng: np.random.Generator
    max_log_weight: float = 50.0
    max_estimated_reward: float = 50.0
    log_weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.n_actions < 2:
            raise ValueError("EXP3 requires at least two actions.")
        self.log_weights = np.zeros((self.n_zones, self.n_actions), dtype=float)

    def reset(self) -> None:
        """Forget learned preferences while retaining this agent's settings/RNG."""
        self.log_weights.fill(0.0)

    def probabilities(self) -> np.ndarray:
        """Return exploration-mixed action probabilities for every zone."""
        centered = self.log_weights - self.log_weights.max(axis=1, keepdims=True)
        base_weights = np.exp(np.clip(centered, -self.max_log_weight, 0.0))
        base_probabilities = base_weights / np.maximum(base_weights.sum(axis=1, keepdims=True), 1e-12)
        probabilities = (1.0 - self.exploration) * base_probabilities + self.exploration / self.n_actions
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def select_actions(self) -> np.ndarray:
        """Sample the selected price-action index for each zone."""
        probabilities = self.probabilities()
        return np.asarray(
            [self.rng.choice(self.n_actions, p=probabilities[zone]) for zone in range(self.n_zones)],
            dtype=int,
        )

    def act(self, observation: MarketObservation) -> PolicyAction:
        """Choose actions through the shared API; EXP3 presently ignores context."""
        del observation
        return PolicyAction(self.select_actions())

    def update(self, actions: np.ndarray, raw_rewards: np.ndarray) -> None:
        """Update selected arms using importance-weighted, normalized revenue."""
        actions = np.asarray(actions, dtype=int)
        rewards = np.nan_to_num(np.asarray(raw_rewards, dtype=float), nan=0.0, posinf=self.reward_normalizer, neginf=0.0)
        scaled_rewards = np.clip(rewards / self.reward_normalizer, 0.0, 1.0)
        probabilities = self.probabilities()
        zones = np.arange(self.n_zones)
        selected_probabilities = np.maximum(probabilities[zones, actions], 1e-12)
        estimates = np.clip(scaled_rewards / selected_probabilities, 0.0, self.max_estimated_reward)
        self.log_weights[zones, actions] += self.learning_rate * estimates

        # Shifting is mathematically neutral and bounds exponentiation later.
        self.log_weights -= self.log_weights.max(axis=1, keepdims=True)
        self.log_weights[:] = np.clip(self.log_weights, -self.max_log_weight, 0.0)

    def observe(self, transition: Transition) -> None:
        """Apply the EXP3 update from one generic market transition."""
        self.update(transition.action.indexes, transition.zone_rewards)

    def end_episode(self) -> None:
        """EXP3 updates step-by-step, so no episode-level work is needed."""

    def snapshot(self) -> PlatformPolicy:
        """Freeze the current stochastic policy without sharing mutable state.

        EXP3's learned policy is its per-zone probability matrix, not its
        expected multiplier. The snapshot copies that matrix and clones the RNG
        state so frozen action sampling cannot change the live agent's future
        exploratory sequence after it is unfrozen.
        """
        frozen_rng = np.random.default_rng()
        frozen_rng.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
        return FrozenZoneExp3(self.probabilities().copy(), frozen_rng)

    def expected_action_values(self, action_values: np.ndarray) -> np.ndarray:
        """Return the expected multiplier/value per zone under current policy."""
        return self.probabilities() @ np.asarray(action_values, dtype=float)


@dataclass
class FrozenZoneExp3:
    """A non-learning snapshot of one EXP3 platform policy.

    This object samples an action from the probability distribution captured at
    freeze time. It intentionally has no weights or update rule: market shocks
    still affect realised revenue, but never the frozen policy.
    """

    action_probabilities: np.ndarray
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.action_probabilities = np.asarray(self.action_probabilities, dtype=float).copy()
        if self.action_probabilities.ndim != 2 or self.action_probabilities.shape[1] < 2:
            raise ValueError("frozen EXP3 probabilities must have shape (n_zones, n_actions).")
        if np.any(self.action_probabilities < 0) or not np.allclose(self.action_probabilities.sum(axis=1), 1.0):
            raise ValueError("each frozen EXP3 probability row must sum to one.")

    def act(self, observation: MarketObservation) -> PolicyAction:
        """Keep sampling prices from the fixed distribution during a freeze."""
        del observation
        n_actions = self.action_probabilities.shape[1]
        indexes = np.asarray(
            [self.rng.choice(n_actions, p=probabilities) for probabilities in self.action_probabilities],
            dtype=int,
        )
        return PolicyAction(indexes)

    def observe(self, transition: Transition) -> None:
        """Deliberately ignore outcomes: a frozen policy cannot learn."""
        del transition

    def end_episode(self) -> None:
        """A frozen EXP3 policy has no deferred learning work."""
