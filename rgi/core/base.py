from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence, Any
import numpy as np

TGame = TypeVar("TGame", bound="Game[Any, Any]")  # pylint: disable=invalid-name
TGameState = TypeVar("TGameState")  # pylint: disable=invalid-name
TPlayerState = TypeVar("TPlayerState")  # pylint: disable=invalid-name
TAction = TypeVar("TAction")  # pylint: disable=invalid-name
TPlayerId = int  # pylint: disable=invalid-name


class Game(ABC, Generic[TGameState, TAction]):
    @abstractmethod
    def initial_state(self) -> TGameState:
        """Create an initial game state."""

    @abstractmethod
    def current_player_id(self, game_state: TGameState) -> TPlayerId:
        """Return the ID of the current player. Sequential starting at 1."""

    @abstractmethod
    def num_players(self, game_state: TGameState) -> int:
        """Number of players in the game."""

    def player_ids(self, game_state: TGameState) -> Sequence[TPlayerId]:
        """Return a sequence of all player IDs in the game."""
        return range(1, self.num_players(game_state) + 1)

    @abstractmethod
    def legal_actions(self, game_state: TGameState) -> Sequence[TAction]:
        """Return a sequence of all legal actions for the game state."""

    @abstractmethod
    def all_actions(self) -> Sequence[TAction] | None:
        """Optionally return a sequence of all possible actions in the game."""

    @abstractmethod
    def next_state(self, game_state: TGameState, action: TAction) -> TGameState:
        """Return a new immutable game state. Must not modify the input state."""

    @abstractmethod
    def is_terminal(self, game_state: TGameState) -> bool:
        """Return True if the game is in a terminal state."""

    @abstractmethod
    def reward(self, game_state: TGameState, player_id: TPlayerId) -> float:
        """Return the reward for the given player in the given state.

        This is typically 0 for non-terminal states, and -1, 0, or 1 for terminal states,
        depending on whether the player lost, drew, or won respectively."""

    def reward_array(self, game_state: TGameState) -> np.ndarray:
        """Return the reward for the given player in the given state as a NumPy array.

        This is typically 0 for non-terminal states, and -1, 0, or 1 for terminal states,
        depending on whether the player lost, drew, or won respectively."""
        return np.array([self.reward(game_state, player_id) for player_id in self.player_ids(game_state)])

    @abstractmethod
    def pretty_str(self, game_state: TGameState) -> str:
        """Return a human-readable string representation of the game state."""


class Player(ABC, Generic[TGameState, TPlayerState, TAction]):
    @abstractmethod
    def select_action(self, game_state: TGameState, legal_actions: Sequence[TAction]) -> TAction:
        """Select an action from the legal actions."""

    def update_player_state(self, old_game_state: TGameState, action: TAction, new_game_state: TGameState) -> None:
        """Update the player's internal state based on the game state and action.

        This method is called after each action, allowing the player to update any
        internal state or learning parameters based on the game progression."""

    def get_player_state(self) -> TPlayerState:
        """Return the player's internal state."""
        return None  # type: ignore


class GameSerializer(ABC, Generic[TGame, TGameState, TAction]):
    """Companion class to Game that serializes game states for various purposes."""

    @abstractmethod
    def serialize_state(self, game: TGame, game_state: TGameState) -> dict[str, Any]:
        """Serialize the game state to a dictionary for frontend consumption.

        Result must be JSON serializable.
        """

    @abstractmethod
    def parse_action(self, game: TGame, action_data: dict[str, Any]) -> TAction:
        """Parse an action from frontend data."""
