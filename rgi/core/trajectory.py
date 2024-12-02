import dataclasses
from pathlib import Path
import typing
from typing import Generic, Sequence, Type, Any, Union

import numpy as np
from numpy.typing import NDArray

from rgi.core import base
from rgi.core.base import TGameState, TAction


def _is_optional(type_hint: Any) -> bool:
    """Check if a type hint is Optional[T]."""
    return typing.get_origin(type_hint) is Union and type(None) in typing.get_args(type_hint)


@dataclasses.dataclass
class GameTrajectory(Generic[TGameState, TAction]):
    """Collection of sequences & actions representing a single game.

    Args:
        game_states: Sequence of game states, starting with initial state and including all intermediate states
        actions: Sequence of actions taken during the game
        action_player_ids: Sequence of player IDs indicating which player took each action
        action_player_states: Sequence of player states at the time each action was taken
        incremental_rewards: Sequence of reward changes for the acting player after each action
        num_players: Total number of players in the game
        final_reward: Final rewards for each player at game end
    """

    game_states: Sequence[TGameState]
    actions: Sequence[TAction]
    action_player_ids: Sequence[int]
    incremental_rewards: Sequence[float]  # Change in reward for current player after their most recent action.
    num_players: int
    final_reward: Sequence[float]

    # validate
    def __post_init__(self) -> None:
        if len(self.game_states) == 0:
            raise ValueError("The 'game_states' list must contain at least one state.")
        if len(self.actions) != len(self.game_states) - 1:
            raise ValueError(
                f"The number of states ({len(self.game_states)}) must be one more than the number of actions "
                f"({len(self.actions)})"
            )
        if len(self.actions) != len(self.action_player_ids):
            raise ValueError(
                f"The number of actions ({len(self.actions)}) must be the same as the number of action player ids "
                f"({len(self.action_player_ids)})"
            )
        if len(self.actions) != len(self.incremental_rewards):
            raise ValueError(
                f"The number of actions ({len(self.actions)}) must be the same as the number of incremental rewards "
                f"({len(self.incremental_rewards)})"
            )
        if len(self.final_reward) != self.num_players:
            raise ValueError(
                f"The number of final rewards ({len(self.final_reward)}) must be the same as the number of players "
                f"({self.num_players})"
            )

    def write(self, file: Path | str | typing.BinaryIO, allow_pickle: bool = False) -> None:
        """Write trajectory to file or streamusing numpy's efficient binary format."""

        def to_valid_array(name: str, seq: Sequence[Any], assert_dtype: type[np.generic] | None = None) -> NDArray[Any]:
            arr = np.asarray(seq)
            if assert_dtype is not None and arr.dtype != assert_dtype:
                raise ValueError(f"Error saving '{name}': Expected dtype {assert_dtype}, got {arr.dtype}")
            if arr.dtype == np.dtype("O") and not allow_pickle:
                raise ValueError(f"Error saving '{name}': Sequence contains object dtype requiring pickle: {seq}")
            return arr

        # Convert sequences to numpy arrays if they aren't already
        action_player_ids = to_valid_array("action_player_ids", self.action_player_ids, np.int64)
        incremental_rewards = to_valid_array("incremental_rewards", self.incremental_rewards, np.float64)
        final_reward = to_valid_array("final_reward", self.final_reward, np.float64)

        # For states and actions, we need to handle the dataclass fields
        state_arrays = {}
        if dataclasses.is_dataclass(self.game_states[0]):
            for field_name in self.game_states[0].__dataclass_fields__:
                values = [getattr(state, field_name) for state in self.game_states]
                state_arrays[field_name] = to_valid_array(field_name, values)
        else:
            state_arrays[""] = to_valid_array("game_states", self.game_states)

        action_arrays = {}
        if dataclasses.is_dataclass(self.actions[0]):
            for field_name in self.actions[0].__dataclass_fields__:
                values = [getattr(action, field_name) for action in self.actions]
                action_arrays[field_name] = to_valid_array(field_name, values)
        else:
            action_arrays[""] = to_valid_array("actions", self.actions)

        # Save all arrays in a single .npz file
        np.savez_compressed(
            file,
            action_player_ids=action_player_ids,
            incremental_rewards=incremental_rewards,
            final_reward=final_reward,
            num_players=to_valid_array("num_players", [self.num_players]),
            **{f"state_{k}": v for k, v in state_arrays.items()},
            **{f"action_{k}": v for k, v in action_arrays.items()},
        )

    @classmethod
    def read(
        cls,
        file: Path | str | typing.BinaryIO,
        game_state_type: Type[TGameState],
        action_type: Type[TAction],
        allow_pickle: bool = False,
    ) -> "GameTrajectory[TGameState, TAction]":
        """Read trajectory from file or stream in numpy binary format."""
        data = np.load(file, allow_pickle=allow_pickle)

        # Reconstruct states
        if dataclasses.is_dataclass(game_state_type):
            game_state_fields = game_state_fields = {
                field_name: data[f"state_{field_name}"] for field_name in game_state_type.__dataclass_fields__
            }
            game_states = [
                game_state_type(**{k: v[i] for k, v in game_state_fields.items()})
                for i in range(len(next(iter(game_state_fields.values()))))
            ]
        else:
            game_states = data["state_"]

        if dataclasses.is_dataclass(action_type):
            action_fields = {
                field_name: data[f"action_{field_name}"] for field_name in action_type.__dataclass_fields__
            }
            actions = [
                action_type(**{k: v[i] for k, v in action_fields.items()})
                for i in range(len(next(iter(action_fields.values()))))
            ]
        else:
            actions = data["action_"]

        return cls(
            game_states=game_states,  # type: ignore
            actions=actions,  # type: ignore
            action_player_ids=data["action_player_ids"],
            incremental_rewards=data["incremental_rewards"],
            num_players=data["num_players"].item(),
            final_reward=data["final_reward"],
        )


class TrajectoryBuilder(Generic[TGameState, TAction]):

    def __init__(self, game: base.Game[TGameState, TAction], initial_state: TGameState):
        self.game = game
        self.num_players: int = game.num_players(initial_state)
        self.states: list[TGameState] = [initial_state]
        self.actions: list[TAction] = []
        self.action_player_ids: list[int] = []
        self.incremental_rewards: list[float] = []
        self.final_reward: list[float] = []

    def record_step(
        self,
        action_player_id: int,
        action: TAction,
        updated_state: TGameState,
        incremental_reward: float,
    ) -> None:
        self.states.append(updated_state)
        self.actions.append(action)
        self.action_player_ids.append(action_player_id)
        self.incremental_rewards.append(incremental_reward)

    def build(self) -> GameTrajectory[TGameState, TAction]:
        final_reward = [self.game.reward(self.states[-1], player_id) for player_id in range(1, self.num_players + 1)]
        return GameTrajectory(
            self.states,
            self.actions,
            self.action_player_ids,
            self.incremental_rewards,
            self.num_players,
            final_reward,
        )
