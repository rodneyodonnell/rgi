This document describes the class design for the "core" classes of the RDI project.

## Class Design Principles

- Game & Action do not store any state and they are never modified after cration.
- TGameState and TPlayerState are immutable. They should usually be based on dataclasses and python's `immutables` library for performance.

## rdi.core classes
```python
from abc import ABC
from typing import Generic, TypeVar

TGameState = TypeVar('TGameState')
TPlayerState = TypeVar('TPlayerState')
TPlayerId = TypeVar('TPlayerId')
TAction = TypeVar('TAction')
TEmbedding = TypeVar('TEmbedding')

class Game(ABC, Generic[TGameState, TPlayerId, TAction]):
    def initial_state(self) -> TGameState:
    def get_current_player(self, state: TGameState) -> TPlayerId:
    def legal_actions(self, state: TGameState) -> list[TAction]:
    def next_state(self, state: TGameState, action: TAction) -> TGameState:
    def is_terminal(self, state: TGameState) -> bool:
    def reward(self, state: TGameState, player_id: TPlayerId) -> float:

class StateEmbedder(ABC, Generic[TGameState, TEmbedding]):
    def embed_state(self, state: TGameState) -> TEmbedding:
    def get_embedding_dim(self) -> int:

class ActionEmbedder(ABC, Generic[TAction, TEmbedding]):
    def embed_action(self, action: TAction) -> TEmbedding:
    def get_embedding_dim(self) -> int:

class Player(ABC, Generic[TGameState, TPlayerState, TAction]):
    def select_action(self, game_state: TGameState, legal_actions: list[TAction]) -> TAction:
    def update_state(self, game_state: TGameState, action: TAction):

class GameObserver(ABC, Generic[TGameState, TPlayerId]):
    def observe_initial_state(self, state: TGameState):
    def observe_action(self, state: TGameState, player: TPlayerId, action: TAction):
    def observe_state_transition(self, old_state: TGameState, new_state: TGameState):
    def observe_game_end(self, final_state: TGameState):
```
