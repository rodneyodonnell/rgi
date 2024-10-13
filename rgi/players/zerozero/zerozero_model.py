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
        self.shared_state_layer: nn.Module = nn.Sequential([nn.Dense(self.shared_dim), nn.relu])
        self.shared_action_layer: nn.Module = nn.Sequential([nn.Dense(self.shared_dim), nn.relu])

        self.dynamics_head: nn.Module = nn.Sequential(
            [nn.Dense(self.hidden_dim), nn.relu, nn.Dense(self.embedding_dim)]
        )
        self.reward_head: nn.Module = nn.Sequential([nn.Dense(self.hidden_dim), nn.relu, nn.Dense(1)])
        self.policy_head: nn.Module = nn.Sequential([nn.Dense(self.hidden_dim), nn.relu, nn.Dense(self.embedding_dim)])

    @nn.compact
    def __call__(self, state: TGameState, action: TAction | None) -> tuple[TEmbedding, jax.Array, TEmbedding]:
        raise NotImplementedError("call compute_next_state or predict_action_policy_and_reward directly.")
        # next_state, reward, policy_embedding = None, None, None
        # if action is None:
        #     next_state = self.compute_next_state(state)
        # else:
        #     next_state, policy_embedding = self.compute_next_state_and_reward(
        #         state, action
        #     )
        # return next_state, reward, policy_embedding

    # @nn.compact
    # def compute_next_state(self, state: TEmbedding, action: TEmbedding) -> TEmbedding:
    #     raise NotImplementedError("call compute_next_state or predict_action_policy_and_reward directly.")
    #     state_embedding = self.state_embedder(state)
    #     state_layer = self.shared_state_layer(state_embedding)
    #     action_embedding = self.action_embedder(action)
    #     action_layer = self.shared_action_layer(action_embedding)

    #     combined_layer = jnp.concatenate([state_layer, action_layer], axis=1)
    #     next_state_embedding = self.dynamics_head(combined_layer)
    #     return next_state_embedding

    @nn.compact
    def predict_action_policy_and_reward(self, state: TGameState) -> tuple[TEmbedding, jax.Array]:
        state_embedding = self.state_embedder(state)
        shared_features = self.shared_state_layer(state_embedding)

        reward = self.reward_head(shared_features).squeeze()  # Remove .item()
        policy_embedding = self.policy_head(shared_features)

        reward = self.reward_head(combined_layer).squeeze()  # Remove .item()
        policy_embedding = self.policy_head(combined_layer)

        return policy_embedding, reward

    # def compute_action_logits(self, policy_embedding: TEmbedding) -> jax.Array:
    #     all_action_embeddings = jnp.array([self.action_embedder(action) for action in self.possible_actions])
    #     return jnp.dot(policy_embedding, jnp.transpose(all_action_embeddings))

    # def compute_action_probabilities(self, policy_embedding: TEmbedding) -> jax.Array:
    #     return jax.nn.softmax(self.compute_action_logits(policy_embedding))

    # def get_state_embedding(self, state: jax.Array) -> TEmbedding:
    #     return self.state_embedder(state)

    # def get_action_embedding(self, action: jax.Array) -> TEmbedding:
    #     return self.action_embedder(action)


def zerozero_loss(
    model: ZeroZeroModel[TGameState, Any, TAction],
    params: dict[str, Any],
    state: TGameState,
    action: TAction,
    next_state: TGameState,
    reward: jax.Array,
    policy_target: jax.Array,
) -> tuple[float, dict[str, float]]:
    next_state_pred: jax.Array
    reward_pred: float
    policy_embedding: jax.Array

    # def compute_next_state(self, state: TEmbedding, action: TEmbedding) -> TEmbedding:
    # def predict_action_policy_and_reward(self, state: TGameState)
    next_state_pred = model.apply(params, state, action, method=model.compute_next_state)  # type: ignore
    reward_pred, policy_embedding = model.apply(params, state, method=model.predict_action_policy_and_reward)  # type: ignore

    # next_state_pred, reward_pred, policy_embedding = model.apply(params, state, action)  # type: ignore

    # next_state_true = model.state_embedder.apply(params, next_state)
    next_state_true = model.apply(params, next_state, method=model.get_state_embedding)
    assert isinstance(next_state_true, jax.Array)

    # Dynamics loss (cosine similarity)
    dynamics_loss = 1 - jnp.sum(next_state_pred * next_state_true, axis=1) / (
        jnp.linalg.norm(next_state_pred, axis=1) * jnp.linalg.norm(next_state_true, axis=1)
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

    total_loss = jnp.mean(dynamics_loss + reward_loss + policy_loss)
    loss_dict = {
        "total_loss": total_loss,
        "dynamics_loss": jnp.mean(dynamics_loss),
        "reward_loss": jnp.mean(reward_loss),
        "policy_loss": jnp.mean(policy_loss),
    }

    return total_loss, loss_dict
