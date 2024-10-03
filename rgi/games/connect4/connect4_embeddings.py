import jax
import jax.numpy as jnp
from flax import linen as nn
from rgi.core.base import StateEmbedder, ActionEmbedder
from rgi.games.connect4.connect4 import Connect4State
from typing import Any, Dict

class Connect4CNN(nn.Module):
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Conv(features=32, kernel_size=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = x.reshape(-1)  # Flatten the output
        x = nn.Dense(features=self.embedding_dim)(x)
        return x

class Connect4StateEmbedder(StateEmbedder[Connect4State, jax.Array]):
    def __init__(self, cnn_model: Connect4CNN):
        self.cnn_model = cnn_model

    def embed_state(self, params: Dict[str, Any], state: Connect4State) -> jax.Array:
        if not isinstance(state, Connect4State):
            raise ValueError("Input must be a Connect4State")
        board_array = self._state_to_array(state)
        board_batch = board_array[jnp.newaxis, ..., jnp.newaxis]
        embedding = self.cnn_model.apply(params['cnn_model'], board_batch)
        if not isinstance(embedding, jax.Array):
            raise TypeError(f"Expected jax.Array, got {type(embedding)}")
        return embedding

    def _state_to_array(self, state: Connect4State) -> jax.Array:
        board_array = jnp.zeros((6, 7), dtype=jnp.float32)
        for (row, col), value in state.board.items():
            board_array = board_array.at[row - 1, col - 1].set(1.0 if value == 1 else -1.0)
        return board_array

    def get_embedding_dim(self) -> int:
        return self.cnn_model.embedding_dim

class Connect4ActionEmbedder(nn.Module):
    embedding_dim: int = 64
    num_actions: int = 7

    @nn.compact
    def __call__(self, action: jax.Array) -> jax.Array:
        action_embeddings = self.param('action_embeddings',
                                       nn.initializers.normal(stddev=0.02),
                                       (self.num_actions, self.embedding_dim))
        return action_embeddings[action]

    def embed_action(self, params: Dict[str, Any], action: int) -> jax.Array:
        if not 1 <= action <= self.num_actions:
            raise ValueError(f"Action must be between 1 and {self.num_actions}")
        embedding = self.apply(params, jnp.array(action - 1))
        if not isinstance(embedding, jax.Array):
            raise TypeError(f"Expected jax.Array, got {type(embedding)}")
        return embedding
    def get_embedding_dim(self) -> int:
        return self.embedding_dim