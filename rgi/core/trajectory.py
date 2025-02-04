import dataclasses
from typing import Generic, Sequence, Any, get_origin, get_args, Union


from rgi.core import base
from rgi.core.base import TGameState, TAction


def _is_optional(type_hint: Any) -> bool:
    """Check if a type hint is Optional[T]."""
    return get_origin(type_hint) is Union and type(None) in get_args(type_hint)


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
