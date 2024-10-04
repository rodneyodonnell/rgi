import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Dict, Optional, Tuple, Generic, Literal
from rgi.core.base import StateEmbedder, ActionEmbedder, TGameState, TPlayerState, TAction

TEmbedding = jax.Array

class ZeroZeroModel(nn.Module, Generic[TGameState, TPlayerState, TAction]):
    state_embedder: StateEmbedder[TGameState, TEmbedding]
    action_embedder: ActionEmbedder[TAction, TEmbedding]
    possible_actions: list[TAction]
    embedding_dim: int = 64
    hidden_dim: int = 128
    shared_dim: int = 256

    def setup(self):
        self.shared_layer: nn.Module = nn.Sequential([
            nn.Dense(self.shared_dim),
            nn.relu
        ])

        self.dynamics_head: nn.Module = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.embedding_dim)
        ])

        self.reward_head: nn.Module = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(1)
        ])

        self.policy_head: nn.Module = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.embedding_dim)
        ])

    @nn.compact
    def __call__(self, state: TGameState, action: TAction) -> tuple[TEmbedding, float, TEmbedding]:
        state_embedding = self.state_embedder.embed_state(state)       
        action_embedding = self.action_embedder.embed_action(action)
        combined_embedding = jnp.concatenate([state_embedding, action_embedding])

        shared_features = self.shared_layer(combined_embedding)

        next_state_embedding = self.dynamics_head(shared_features)
        reward = self.reward_head(shared_features).squeeze().item()
        policy_embedding = self.policy_head(shared_features)

        return next_state_embedding, reward, policy_embedding

    def compute_action_probabilities(self, policy_embedding: TEmbedding) -> TEmbedding:
        # TODO: Precompute this?
        all_action_embeddings = jnp.array([self.action_embedder.embed_action(action) for action in self.possible_actions])
        logits = jnp.dot(all_action_embeddings, policy_embedding)
        return jax.nn.softmax(logits)

def zerozero_loss(params: Dict[str, Any], model: ZeroZeroModel[TGameState, TPlayerState, TAction], 
                  state: TGameState, action: TAction, next_state: TGameState, 
                  reward: float, policy_target: jax.Array) -> Tuple[float, Dict[str, float]]:
    next_state_pred: jax.Array
    reward_pred: float
    policy_embedding: jax.Array
    next_state_pred, reward_pred, policy_embedding = model.apply(params, state, action) # type: ignore
    next_state_true = model.state_embedder.embed_state(next_state)

    # Dynamics loss (cosine similarity)
    dynamics_loss = 1 - jnp.dot(next_state_pred, next_state_true) / (jnp.linalg.norm(next_state_pred) * jnp.linalg.norm(next_state_true))

    # Reward loss (MSE)
    reward_loss = jnp.mean((reward_pred - reward) ** 2)

    # Policy loss (cross-entropy)
    action_probs = model.compute_action_probabilities(policy_embedding)
    policy_loss = -jnp.sum(policy_target * jnp.log(action_probs + 1e-8))

    total_loss = dynamics_loss + reward_loss + policy_loss
    loss_dict = {
        'total_loss': total_loss,
        'dynamics_loss': dynamics_loss,
        'reward_loss': reward_loss,
        'policy_loss': policy_loss
    }

    return total_loss, loss_dict