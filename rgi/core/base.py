from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence, Any
import torch
import torch.nn as nn

TGame = TypeVar("TGame", bound="Game[Any, Any, Any]")  # pylint: disable=invalid-name
TGameState = TypeVar("TGameState")  # pylint: disable=invalid-name
TPlayerState = TypeVar("TPlayerState")  # pylint: disable=invalid-name
TPlayerId = TypeVar("TPlayerId")  # pylint: disable=invalid-name
TAction = TypeVar("TAction")  # pylint: disable=invalid-name
TEmbedding = TypeVar("TEmbedding", bound=torch.Tensor)  # pylint: disable=invalid-name
TParams = TypeVar("TParams")  # pylint: disable=invalid-name


class Game(ABC, Generic[TGameState, TPlayerId, TAction]):
    @abstractmethod
    def initial_state(self) -> TGameState:
        pass

    @abstractmethod
    def current_player_id(self, game_state: TGameState) -> TPlayerId:
        pass

    @abstractmethod
    def all_player_ids(self, game_state: TGameState) -> Sequence[TPlayerId]:
        """Return a sequence of all player IDs in the game."""

    @abstractmethod
    def legal_actions(self, game_state: TGameState) -> Sequence[TAction]:
        pass

    @abstractmethod
    def all_actions(self) -> Sequence[TAction] | None:
        """Optionally return a sequence of all possible actions in the game."""

    @abstractmethod
    def next_state(self, game_state: TGameState, action: TAction) -> TGameState:
        pass

    @abstractmethod
    def is_terminal(self, game_state: TGameState) -> bool:
        pass

    @abstractmethod
    def reward(self, game_state: TGameState, player_id: TPlayerId) -> float:
        """Return the reward for the given player in the given state.

        This is typically 0 for non-terminal states, and -1, 0, or 1 for terminal states,
        depending on whether the player lost, drew, or won respectively."""

    @abstractmethod
    def pretty_str(self, game_state: TGameState) -> str:
        """Return a human-readable string representation of the game state."""


class GameSerializer(ABC, Generic[TGame, TGameState, TAction]):
    """Companion class to Game that serializes game states for various purposes."""

    @abstractmethod
    def serialize_state(self, game: TGame, game_state: TGameState) -> dict[str, Any]:
        """Serialize the game state to a dictionary for frontend consumption."""

    @abstractmethod
    def parse_action(self, game: TGame, action_data: dict[str, Any]) -> TAction:
        """Parse an action from frontend data."""


class StateEmbedder(nn.Module, Generic[TGameState]):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, game_states: TGameState) -> torch.Tensor:
        raise NotImplementedError


class ActionEmbedder(nn.Module, Generic[TAction]):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, game_actions: TAction) -> torch.Tensor:
        raise NotImplementedError

    def all_action_embeddings(self) -> torch.Tensor:
        raise NotImplementedError


class Player(ABC, Generic[TGameState, TPlayerState, TAction]):
    @abstractmethod
    def select_action(self, game_state: TGameState, legal_actions: Sequence[TAction]) -> TAction:
        pass

    @abstractmethod
    def update_state(self, game_state: TGameState, action: TAction) -> None:
        """Update the player's internal state based on the game state and action.

        This method is called after each action, allowing the player to update any
        internal state or learning parameters based on the game progression."""
