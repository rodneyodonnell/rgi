from typing import Any, List, Tuple
from typing_extensions import override
import torch

from rgi.core.base import Game, GameSerializer, TGameState, TPlayerId, TAction


class Count21Game(Game[Tuple[int, ...], int, int]):
    def __init__(self, target: int = 21):
        self.target = target

    @override
    def initial_state(self) -> Tuple[int, ...]:
        return (0,)

    @override
    def current_player_id(self, state: Tuple[int, ...]) -> int:
        return 2 - len(state) % 2

    @override
    def all_player_ids(self, state: Tuple[int, ...]) -> List[int]:
        return [1, 2]

    @override
    def legal_actions(self, state: Tuple[int, ...]) -> List[int]:
        return [1, 2, 3]

    @override
    def all_actions(self) -> List[int]:
        return [1, 2, 3]

    @override
    def next_state(self, state: Tuple[int, ...], action: int) -> Tuple[int, ...]:
        return state + (action,)

    @override
    def is_terminal(self, state: Tuple[int, ...]) -> bool:
        return sum(state) >= self.target

    @override
    def reward(self, state: Tuple[int, ...], player_id: int) -> float:
        if not self.is_terminal(state):
            return 0.0
        return 1.0 if self.current_player_id(state) == player_id else -1.0

    @override
    def pretty_str(self, state: Tuple[int, ...]) -> str:
        return f"Count: {sum(state)}, Moves: {state}"


class Count21Serializer(GameSerializer[Count21Game, Tuple[int, ...], int]):
    @override
    def serialize_state(self, game: Count21Game, state: Tuple[int, ...]) -> dict[str, Any]:
        return {"state": state}

    @override
    def parse_action(self, game: Count21Game, action_data: dict[str, Any]) -> int:
        return action_data["action"]

    @override
    def state_to_tensor(self, game: Count21Game, state: Tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.long)

    @override
    def action_to_tensor(self, game: Count21Game, action: int) -> torch.Tensor:
        return torch.tensor(action, dtype=torch.long)

    @override
    def tensor_to_action(self, game: Count21Game, action_tensor: torch.Tensor) -> int:
        return int(action_tensor.item())

    @override
    def tensor_to_state(self, game: Count21Game, state_tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(state_tensor.tolist())
