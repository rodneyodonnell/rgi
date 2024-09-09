from abc import ABC, abstractmethod
from typing import Generic, TypeVar

State = TypeVar('State')
Action = TypeVar('Action')
Player = TypeVar('Player')

class Game(ABC, Generic[State, Action, Player]):
    @abstractmethod
    def initial_state(self) -> State:
        """Return the initial state of the game."""
        pass

    @abstractmethod
    def current_player(self, state: State) -> Player:
        """Return the player whose turn it is in the given state."""
        pass

    @abstractmethod
    def legal_actions(self, state: State) -> list[Action]:
        """Return a list of legal actions for the current player in the given state."""
        pass

    @abstractmethod
    def next_state(self, state: State, action: Action) -> State:
        """Return the state that results from taking the given action in the given state."""
        pass

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Return True if the given state is terminal (game over), False otherwise."""
        pass

    @abstractmethod
    def reward(self, state: State, player: Player) -> float:
        """Return the reward for the given player in the given state."""
        pass

    @abstractmethod
    def action_to_string(self, action: Action) -> str:
        """Convert an action to a string representation."""
        pass

    @abstractmethod
    def state_to_string(self, state: State) -> str:
        """Convert a state to a string representation."""
        pass

    @abstractmethod
    def string_to_action(self, string: str) -> Action:
        """Convert a string representation to an action."""
        pass