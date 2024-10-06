from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any

TGame = TypeVar("TGame", bound="Game[Any, Any, Any]")  # pylint: disable=invalid-name
TGameState = TypeVar("TGameState")  # pylint: disable=invalid-name
TPlayerState = TypeVar("TPlayerState")  # pylint: disable=invalid-name
TPlayerId = TypeVar("TPlayerId")  # pylint: disable=invalid-name
TAction = TypeVar("TAction")  # pylint: disable=invalid-name
TEmbedding = TypeVar("TEmbedding")  # pylint: disable=invalid-name
TParams = TypeVar("TParams")  # pylint: disable=invalid-name


class Game(ABC, Generic[TGameState, TPlayerId, TAction]):
    @abstractmethod
    def initial_state(self) -> TGameState:
        pass

    @abstractmethod
    def current_player_id(self, state: TGameState) -> TPlayerId:
        pass

    @abstractmethod
    def all_player_ids(self, state: TGameState) -> list[TPlayerId]:
        """Return a sequence of all player IDs in the game."""

    @abstractmethod
    def legal_actions(self, state: TGameState) -> list[TAction]:
        pass

    @abstractmethod
    def all_actions(self) -> list[TAction] | None:
        """Optionally return a list of all possible actions in the game."""

    @abstractmethod
    def next_state(self, state: TGameState, action: TAction) -> TGameState:
        pass

    @abstractmethod
    def is_terminal(self, state: TGameState) -> bool:
        pass

    @abstractmethod
    def reward(self, state: TGameState, player_id: TPlayerId) -> float:
        """Return the reward for the given player in the given state.

        This is typically 0 for non-terminal states, and -1, 0, or 1 for terminal states,
        depending on whether the player lost, drew, or won respectively."""

    @abstractmethod
    def pretty_str(self, state: TGameState) -> str:
        """Return a human-readable string representation of the game state."""


class GameSerializer(ABC, Generic[TGame, TGameState, TAction]):
    """Companion class to Game that serializes game states for frontend consumption."""

    @abstractmethod
    def serialize_state(self, game: TGame, state: TGameState) -> dict[str, Any]:
        """Serialize the game state to a dictionary for frontend consumption."""

    @abstractmethod
    def parse_action(self, game: TGame, action_data: dict[str, Any]) -> TAction:
        """Parse an action from frontend data."""


class Player(ABC, Generic[TGameState, TPlayerState, TAction]):
    @abstractmethod
    def select_action(self, game_state: TGameState, legal_actions: list[TAction]) -> TAction:
        pass

    @abstractmethod
    def update_state(self, game_state: TGameState, action: TAction) -> None:
        """Update the player's internal state based on the game state and action.

        This method is called after each action, allowing the player to update any
        internal state or learning parameters based on the game progression."""


class GameObserver(ABC, Generic[TGameState, TPlayerId, TAction]):
    @abstractmethod
    def observe_initial_state(self, state: TGameState) -> None:
        pass

    @abstractmethod
    def observe_action(self, state: TGameState, player: TPlayerId, action: TAction) -> None:
        pass

    @abstractmethod
    def observe_state_transition(self, old_state: TGameState, new_state: TGameState) -> None:
        pass

    @abstractmethod
    def observe_game_end(self, final_state: TGameState) -> None:
        pass
