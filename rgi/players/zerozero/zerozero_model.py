from typing import Any, Generic
from flax.typing import FrozenVariableDict
import jax
import jax.numpy as jnp
from flax import linen as nn
from rgi.core.base import (
    TGameState,
    TPlayerState,
    TAction,
)

# pylint: disable=attribute-defined-outside-init  # vars are defined in setup() for flax code.

TEmbedding = jax.Array
TJaxParams = dict[str, Any]


class StateEmbedder(Generic[TGameState], nn.Module):
    embedding_dim: int

    @nn.compact
    def __call__(self, state: TGameState) -> jax.Array:
        raise NotImplementedError("StateEmbedder must implement __call__ method.")


class ActionEmbedder(Generic[TAction], nn.Module):
    embedding_dim: int

    @nn.compact
    def __call__(self, action: TAction) -> jax.Array:
        raise NotImplementedError("ActionEmbedder must implement __call__ method.")


class ZeroZeroModel(Generic[TGameState, TPlayerState, TAction], nn.Module):
    state_embedder: StateEmbedder[TGameState]
    action_embedder: ActionEmbedder[TAction]
    possible_actions: list[TAction]
    embedding_dim: int = 64
    hidden_dim: int = 128
    shared_dim: int = 256

    def setup(self) -> None:
        self.shared_layer: nn.Module = nn.Sequential([nn.Dense(self.shared_dim), nn.relu])
        self.dynamics_head: nn.Module = nn.Sequential(
            [nn.Dense(self.hidden_dim), nn.relu, nn.Dense(self.embedding_dim)]
        )
        self.reward_head: nn.Module = nn.Sequential([nn.Dense(self.hidden_dim), nn.relu, nn.Dense(1)])
        self.policy_head: nn.Module = nn.Sequential([nn.Dense(self.hidden_dim), nn.relu, nn.Dense(self.embedding_dim)])

    @nn.compact
    def __call__(self, state: TGameState, action: TAction | None) -> tuple[TEmbedding, float, TEmbedding]:
        state_embedding = self.state_embedder(state)
        action_embedding = (
            self.action_embedder(action) if action is not None else jnp.zeros(self.action_embedder.embedding_dim)
        )
        combined_embedding = jnp.concatenate([state_embedding, action_embedding])

        shared_features = self.shared_layer(combined_embedding)

        next_state_embedding = self.dynamics_head(shared_features)
        reward = self.reward_head(shared_features).squeeze().item()
        policy_embedding = self.policy_head(shared_features)

        return next_state_embedding, reward, policy_embedding

    def compute_action_probabilities(self, policy_embedding: TEmbedding) -> TEmbedding:
        all_action_embeddings = jnp.array([self.action_embedder(action) for action in self.possible_actions])
        logits = jnp.dot(all_action_embeddings, policy_embedding)
        return jax.nn.softmax(logits)


def zerozero_loss(
    model: ZeroZeroModel[TGameState, Any, TAction],
    params: dict[str, Any],
    state: TGameState,
    action: TAction,
    next_state: TGameState,
    reward: float,
    policy_target: jax.Array,
) -> tuple[float, dict[str, float]]:
    next_state_pred: jax.Array
    reward_pred: float
    policy_embedding: jax.Array
    next_state_pred, reward_pred, policy_embedding = model.apply(params, state, action)  # type: ignore
    next_state_true = model.state_embedder.apply(params, next_state)
    assert isinstance(next_state_true, jax.Array)

    # Dynamics loss (cosine similarity)
    dynamics_loss = 1 - jnp.dot(next_state_pred, next_state_true) / (
        jnp.linalg.norm(next_state_pred) * jnp.linalg.norm(next_state_true)
    )

    # Reward loss (MSE)
    reward_loss = jnp.mean((reward_pred - reward) ** 2)

    # Policy loss (cross-entropy)
    # action_probs = model.compute_action_probabilities(policy_embedding)
    action_probs = model.apply(
        params,
        method=model.compute_action_probabilities,
        policy_embedding=policy_embedding,
    )
    assert isinstance(action_probs, jax.Array)
    policy_loss = -jnp.sum(policy_target * jnp.log(action_probs + 1e-8))

    total_loss = dynamics_loss + reward_loss + policy_loss
    loss_dict = {
        "total_loss": total_loss,
        "dynamics_loss": dynamics_loss,
        "reward_loss": reward_loss,
        "policy_loss": policy_loss,
    }

    return total_loss, loss_dict
