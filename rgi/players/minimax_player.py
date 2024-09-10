from typing import TypeVar, Any, Optional
from typing_extensions import override
from rgi.core.base import Game, Player

# Defining the TypeVars to match the requested format
TGameState = TypeVar("TGameState")
TAction = TypeVar("TAction")
TPlayerId = TypeVar("TPlayerId")


class MinimaxPlayer(Player[TGameState, Any, TAction]):
    def __init__(self, game: Game[TGameState, TPlayerId, TAction], player_id: TPlayerId, max_depth: int = 4):
        self.game = game
        self.player_id = player_id
        self.max_depth = max_depth

    def evaluate(self, state: TGameState) -> float:
        """Evaluate terminal state. If the game is terminal, return the reward."""
        if self.game.is_terminal(state):
            reward = self.game.reward(state, self.player_id)
            return reward if reward != 0 else -1  # Returning -1 for a loss, 1 for a win, and 0 for draws
        return self.heuristic(state)  # Use heuristic evaluation for non-terminal states

    def heuristic(self, state: TGameState) -> float:
        """Heuristic evaluation for non-terminal states."""
        # return state  # Replace with actual heuristic logic suitable for your game
        return 0

    def minimax(self, state: TGameState, depth: int, maximizing_player: bool) -> tuple[int, Optional[TAction]]:
        """Minimax algorithm with depth limiting."""
        if self.game.is_terminal(state) or depth == 0:
            return self.evaluate(state), None

        legal_actions = self.game.legal_actions(state)

        if maximizing_player:
            max_eval = -float("inf")
            best_action = None
            for action in legal_actions:
                next_state = self.game.next_state(state, action)
                eval_score, _ = self.minimax(next_state, depth - 1, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
            return max_eval, best_action
        else:
            min_eval = float("inf")
            best_action = None
            for action in legal_actions:
                next_state = self.game.next_state(state, action)
                eval_score, _ = self.minimax(next_state, depth - 1, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
            return min_eval, best_action

    @override
    def select_action(self, game_state: TGameState, legal_actions: list[TAction]) -> TAction:
        """Select the best action using the Minimax algorithm."""
        _, best_action = self.minimax(game_state, self.max_depth, True)
        return best_action

    @override
    def update_state(self, game_state: TGameState, action: TAction) -> None:
        """No need for internal updates for this player."""
        pass  # No need for internal updates for this player
