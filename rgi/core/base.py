from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TGameState = TypeVar('TGameState')
TPlayerState = TypeVar('TPlayerState')
TPlayerId = TypeVar('TPlayerId')
TAction = TypeVar('TAction')
TEmbedding = TypeVar('TEmbedding')

class Game(ABC, Generic[TGameState, TPlayerId, TAction]):
    @abstractmethod
    def initial_state(self) -> TGameState:
        pass

    @abstractmethod
    def get_current_player(self, state: TGameState) -> TPlayerId:
        pass

    @abstractmethod
    def legal_actions(self, state: TGameState) -> list[TAction]:
        pass

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
        pass

class StateEmbedder(ABC, Generic[TGameState, TEmbedding]):
    @abstractmethod
    def embed_state(self, state: TGameState) -> TEmbedding:
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        pass

class ActionEmbedder(ABC, Generic[TAction, TEmbedding]):
    @abstractmethod
    def embed_action(self, action: TAction) -> TEmbedding:
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        pass

class Player(ABC, Generic[TGameState, TPlayerState, TAction]):
    @abstractmethod
    def select_action(self, game_state: TGameState, legal_actions: list[TAction]) -> TAction:
        pass

    @abstractmethod
    def update_state(self, game_state: TGameState, action: TAction):
        """Update the player's internal state based on the game state and action.
        
        This method is called after each action, allowing the player to update any
        internal state or learning parameters based on the game progression."""
        pass

class GameObserver(ABC, Generic[TGameState, TPlayerId]):
    @abstractmethod
    def observe_initial_state(self, state: TGameState):
        pass

    @abstractmethod
    def observe_action(self, state: TGameState, player: TPlayerId, action: TAction):
        pass

    @abstractmethod
    def observe_state_transition(self, old_state: TGameState, new_state: TGameState):
        pass

    @abstractmethod
    def observe_game_end(self, final_state: TGameState):
        pass