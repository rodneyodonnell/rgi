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


class Count21Game(base.Game[GameState, PlayerId, Action]):
    def __init__(self, num_players: int = 2, target: int = 21, max_guess: int = 3):
        self.num_players = num_players
        self.target = target
        self._all_actions = tuple(Action(g + 1) for g in range(max_guess))

    @override
    def initial_state(self) -> GameState:
        return GameState(score=0, current_player=1)

    @override
    def current_player_id(self, game_state: GameState) -> PlayerId:
        return game_state.current_player

    @override
    def all_player_ids(self, game_state: GameState) -> Sequence[PlayerId]:
        return range(1, self.num_players + 1)

    @override
    def legal_actions(self, game_state: GameState) -> Sequence[Action]:
        return self._all_actions

    @override
    def all_actions(self) -> Sequence[Action]:
        return self._all_actions

    @override
    def next_state(self, game_state: GameState, action: Action) -> GameState:
        next_player = game_state.current_player % self.num_players + 1
        return GameState(score=game_state.score + action, current_player=next_player)

    @override
    def is_terminal(self, game_state: GameState) -> bool:
        return game_state.score >= self.target

    @override
    def reward(self, game_state: GameState, player_id: PlayerId) -> float:
        if not self.is_terminal(game_state):
            return 0.0
        return 1.0 if self.current_player_id(game_state) == player_id else -1.0

    @override
    def pretty_str(self, game_state: GameState) -> str:
        return f"Score: {game_state.score}, Player: {game_state.current_player}"


class Count21Serializer(base.GameSerializer[Count21Game, GameState, Action]):
    @override
    def serialize_state(self, game: Count21Game, game_state: GameState) -> dict[str, Any]:
        return {"score": game_state.score, "current_player": game_state.current_player}

    @override
    def parse_action(self, game: Count21Game, action_data: dict[str, Any]) -> Action:
        return Action(action_data["action"])


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
