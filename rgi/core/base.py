from typing import Generic, TypeVar, Sequence, Any
from dataclasses import fields, dataclass

from abc import ABC, abstractmethod

TGame = TypeVar("TGame", bound="Game[Any, Any]")  # pylint: disable=invalid-name
TGameState = TypeVar("TGameState")  # pylint: disable=invalid-name
TPlayerState = TypeVar("TPlayerState")  # pylint: disable=invalid-name
TAction = TypeVar("TAction")  # pylint: disable=invalid-name
TPlayerId = int  # pylint: disable=invalid-name

import torch
from torch import nn

TBatchGameState = TypeVar("TBatchGameState", bound="Batchable[Any]")  # pylint: disable=invalid-name
TBatchAction = TypeVar("TBatchAction", bound="Batchable[Any]")  # pylint: disable=invalid-name
TEmbedding = TypeVar("TEmbedding", bound=torch.Tensor)  # pylint: disable=invalid-name

T = TypeVar("T")
TBatch = TypeVar("TBatch", bound="Batch[Any]")


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


class Batchable(Protocol[T]):
    """Protocol to convert single states & actions into torch.Tensor for batching."""

    @staticmethod
    def from_sequence(items: Sequence[T]) -> "Batchable[T]": ...

    def __getitem__(self, index: int) -> T: ...

    def __len__(self) -> int: ...


class Batch(Generic[T]):
    """Convenience class to convert a sequence of states & actions into a batch.

    >>> from dataclasses import dataclass
    >>> import torch
    >>> @dataclass
    ... class GameState:
    ...     score: int
    ...     current_player: int
    >>> @dataclass
    ... class BatchGameState(Batch[GameState]):
    ...     score: torch.Tensor
    ...     current_player: torch.Tensor
    >>> states = [GameState(5, 1), GameState(7, 2)]
    >>> batch = BatchGameState.from_sequence(states)
    >>> len(batch)
    2
    >>> batch
    BatchGameState(score=tensor([5, 7]), current_player=tensor([1, 2]))
    >>> batch[0]
    GameState(score=5, current_player=1)
    """

    _unbatch_class: Type[T]

    @classmethod
    def from_sequence(cls: Type[TBatch], items: Sequence[T]) -> TBatch:
        if not items:
            raise ValueError("Cannot create a batch from an empty sequence")

        cls_fields = set(f.name for f in fields(cls))  # type: ignore
        batch_dict = {}
        for field in fields(items[0]):  # type: ignore
            if field.name not in cls_fields:
                continue
            values = [getattr(item, field.name) for item in items]
            # We need to handle both primitive values and torch.Tensors here.
            # torch.tensor(primitive_list) is probably more efficient, but doesn't work for tensors.
            batch_dict[field.name] = torch.stack([torch.tensor(value) for value in values])

        batch = cls(**batch_dict)
        batch._unbatch_class = type(items[0])
        return batch

    def __getitem__(self, index: int) -> T:
        item_dict = {field.name: field.type(getattr(self, field.name)[index]) for field in fields(self)}  # type: ignore
        return self._unbatch_class(**item_dict)

    def __len__(self) -> int:
        return len(getattr(self, fields(self)[0].name))  # type: ignore


@dataclass
class PrimitiveBatch(Generic[T]):
    """A batch class for primitive types like int, float, etc.

    >>> batch = PrimitiveBatch.from_sequence([2,4,6,8])
    >>> len(batch)
    4
    >>> batch
    PrimitiveBatch(values=tensor([2, 4, 6, 8]))
    >>> batch[0]
    2
    """

    values: torch.Tensor

    @classmethod
    def from_sequence(cls: Type["PrimitiveBatch[T]"], items: Sequence[T]) -> "PrimitiveBatch[T]":
        if not items:
            raise ValueError("Cannot create a batch from an empty sequence")

        return cls(values=torch.tensor(items))

    def __getitem__(self, index: int) -> T:
        return self.values[index].item()  # type: ignore

    def __len__(self) -> int:
        return self.values.shape[0]


class StateEmbedder(ABC, nn.Module, Generic[TBatchGameState]):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, game_states: TBatchGameState) -> torch.Tensor:
        pass


class ActionEmbedder(ABC, nn.Module, Generic[TBatchAction]):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, game_actions: TBatchAction) -> torch.Tensor:
        pass

    @abstractmethod
    def all_action_embeddings(self) -> torch.Tensor:
        pass
