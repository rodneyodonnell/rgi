from typing import Any
from typing_extensions import override

import jax
import jax.numpy as jnp
from flax import linen as nn

from rgi.players.zerozero.zerozero_model import StateEmbedder, ActionEmbedder
from rgi.games.connect4.connect4 import Connect4State, TAction


class Connect4StateEmbedder(StateEmbedder[Connect4State]):
    embedding_dim: int = 64

    def _state_to_array(self, state: Connect4State) -> jax.Array:
        if not isinstance(state, Connect4State):
            raise ValueError("Invalid state type")
        board_array = jnp.zeros((6, 7), dtype=jnp.float32)
        for (row, col), value in state.board.items():
            board_array = board_array.at[row - 1, col - 1].set(1.0 if value == 1 else -1.0)
        return board_array

    @nn.compact
    def __call__(self, state: Connect4State) -> jax.Array:
        x = self._state_to_array(state)
        x = x[..., None]  # Add channel dimensions
        x = nn.Conv(features=32, kernel_size=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = x.reshape(-1)  # Flatten the output
        x = nn.Dense(features=self.embedding_dim)(x)
        return x


class Connect4ActionEmbedder(ActionEmbedder[TAction], nn.Module):
    embedding_dim: int = 64
    num_actions: int = 7

    @nn.compact
    def __call__(self, action: TAction) -> jax.Array:
        if not 1 <= action <= self.num_actions:
            raise ValueError(f"Action must be between 1 and {self.num_actions}")
        action_embeddings = self.param(
            "action_embeddings",
            nn.initializers.normal(stddev=0.02),
            (self.num_actions, self.embedding_dim),
        )
        return action_embeddings[action - 1]
