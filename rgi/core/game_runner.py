from typing import Generic, Sequence

from rgi.core import base
from rgi.core.base import TAction, TGameState, TPlayerData, TPlayerState
from rgi.core.trajectory import GameTrajectory, TrajectoryBuilder


class GameRunner(Generic[TGameState, TAction, TPlayerState, TPlayerData]):
    """Class for running games and recording trajectories."""

    def __init__(
        self,
        game: base.Game[TGameState, TAction],
        players: Sequence[base.Player[TGameState, TPlayerState, TAction, TPlayerData]],
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

        self.trajectory_builder: TrajectoryBuilder[TGameState, TAction, TPlayerData] = TrajectoryBuilder(
            game, self.game_state
        )
        self.verbose = verbose

    def run_step(self) -> None:
        # Record pre-action state
        old_game_state = self.game_state
        old_player_id = self.current_player_id
        old_player_expected_reward = self.game.reward(self.game_state, self.current_player_id)

        # Determine Action
        current_player = self.players[self.current_player_id - 1]
        legal_actions = self.game.legal_actions(old_game_state)
        action_result = current_player.select_action(old_game_state, legal_actions)

        # Calculate & update states.
        updated_game_state = self.game.next_state(old_game_state, action_result.action)
        # Incremental reward for player who just took an action.
        old_player_updated_expected_reward = self.game.reward(updated_game_state, old_player_id)
        incremental_reward = old_player_updated_expected_reward - old_player_expected_reward

        # Update game & player states.
        self.game_state = updated_game_state
        self.current_player_id = self.game.current_player_id(updated_game_state)
        for _player in self.players:
            _player.update_player_state(old_game_state, action_result.action, updated_game_state)

        self.trajectory_builder.record_step(
            old_player_id,
            action_result.action,
            updated_game_state,
            incremental_reward,
            action_result.player_data,
        )

        if self.verbose:
            print(
                f"\nplayer_id={old_player_id}, "
                f"action={action_result.action}, "
                f"player_data={action_result.player_data}, "
                f"next state:\n{self.game.pretty_str(updated_game_state)}\n"
                f"result={self.game.reward(updated_game_state, 1)},{self.game.reward(updated_game_state, 2)}"
            )

    def run(self) -> GameTrajectory[TGameState, TAction, TPlayerData]:
        while not self.game.is_terminal(self.game_state):
            self.run_step()

        return self.trajectory_builder.build()
