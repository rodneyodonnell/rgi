from typing import Generic, Sequence, Any, Union

from rgi.core import base
from rgi.core.base import TGameState, TAction, TPlayerState
from rgi.core.trajectory import GameTrajectory, TrajectoryBuilder


class GameRunner(Generic[TGameState, TAction, TPlayerState]):
    def __init__(
        self,
        game: base.Game[TGameState, TAction],
        players: Sequence[base.Player[TGameState, TPlayerState, TAction]],
        initial_state: TGameState | None = None,
        verbose: bool = False,
    ):
        initial_state = initial_state if initial_state is not None else game.initial_state()

        # constants.
        self.game = game
        self.players = players
        self.num_players = game.num_players(initial_state)

        # Initialize board.
        self.game_state = initial_state
        self.current_player_id = game.current_player_id(self.game_state)

        self.trajectory_builder: TrajectoryBuilder[TGameState, TAction] = TrajectoryBuilder(game, self.game_state)
        self.verbose = verbose

    def run_step(self) -> None:
        # Record pre-action state
        old_game_state = self.game_state
        old_player_id = self.current_player_id
        old_player_expected_reward = self.game.reward(self.game_state, self.current_player_id)

        # Determine Action
        current_player = self.players[self.current_player_id - 1]
        legal_actions = self.game.legal_actions(old_game_state)
        action_result: Union[TAction, tuple[TAction, dict[TAction, int]]] = current_player.select_action(
            old_game_state, legal_actions
        )

        # Handle both simple actions and actions with MCTS policy counts
        if isinstance(action_result, tuple):
            action, mcts_policy_counts = action_result
        else:
            action = action_result
            mcts_policy_counts = None

        # Calculate & update states.
        updated_game_state = self.game.next_state(old_game_state, action)
        # Incremental reward for player who just took an action.
        old_player_updated_expected_reward = self.game.reward(updated_game_state, old_player_id)
        incremental_reward = old_player_updated_expected_reward - old_player_expected_reward

        # Update game & player states.
        self.game_state = updated_game_state
        self.current_player_id = self.game.current_player_id(updated_game_state)
        for _player in self.players:
            _player.update_player_state(old_game_state, action, updated_game_state)

        self.trajectory_builder.record_step(
            old_player_id,
            action,
            updated_game_state,
            incremental_reward,
            mcts_policy_counts,
        )

        if self.verbose:
            print(
                f"\nplayer_id={old_player_id}, "
                f"action={action}, "
                f"next state:\n{self.game.pretty_str(updated_game_state)}\n"
                f"result={self.game.reward(updated_game_state, 1)},{self.game.reward(updated_game_state, 2)}"
            )

    def run(self) -> GameTrajectory[TGameState, TAction]:
        while not self.game.is_terminal(self.game_state):
            self.run_step()

        return self.trajectory_builder.build()
