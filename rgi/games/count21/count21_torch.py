from dataclasses import dataclass

from typing import Sequence, Any
from typing_extensions import override

import torch

from rgi.core import base


@dataclass
class GameState:
    score: int
    current_player: int


@dataclass
class BatchGameState(base.Batch[GameState]):
    score: torch.Tensor
    current_player: torch.Tensor


Action = int
BatchAction = base.PrimitiveBatch[Action]
PlayerId = int


class Count21StateEmbedder(base.StateEmbedder[BatchGameState]):
    def __init__(self, game: Count21Game, embedding_dim: int = 64):
        super().__init__(embedding_dim)
        self.game = game
        self.score_embedding = torch.nn.Embedding(game.target, embedding_dim)
        self.player_embedding = torch.nn.Embedding(game.num_players, embedding_dim)

    def forward(self, game_states: BatchGameState) -> torch.Tensor:
        score_embeddings: torch.Tensor = self.score_embedding(game_states.score)
        player_embeddings: torch.Tensor = self.player_embedding(game_states.current_player)
        return (score_embeddings + player_embeddings) / 2


class Count21ActionEmbedder(base.ActionEmbedder[BatchAction]):
    def __init__(self, game: Count21Game, embedding_dim: int = 64):
        super().__init__(embedding_dim)
        self.game = game
        self.embedding = torch.nn.Embedding(len(self.game.all_actions()), embedding_dim)

    def forward(self, game_actions: BatchAction) -> torch.Tensor:
        action_embeddings: torch.Tensor = self.embedding(game_actions.values)
        return action_embeddings

    def all_action_embeddings(self) -> torch.Tensor:
        return self.embedding.weight[1:]
