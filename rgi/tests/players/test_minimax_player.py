import unittest
from rgi.core.base import Game
from rgi.players.minimax_player import MinimaxPlayer


class TestMinimaxPlayerLookahead(unittest.TestCase):

    def setUp(self):
        # Mock game object for testing
        self.game = self.MockGame()
        self.player1 = MinimaxPlayer(self.game, player_id=1, max_depth=2)
        self.player2 = MinimaxPlayer(self.game, player_id=2, max_depth=2)

    class MockGame(Game):
        def initial_state(self):
            return 0  # Simplified state

        def current_player_id(self, state: int):
            return 1 if state % 2 == 0 else 2

        def all_player_ids(self, state: int):
            return [1, 2]

        def legal_actions(self, state: int):
            return [1, 2]  # Two actions

        def next_state(self, state: int, action: int):
            return state + action  # Simplified state progression

        def is_terminal(self, state: int):
            return state >= 3  # Terminal state when the state reaches 3

        def reward(self, state: int, player_id: int):
            # Player 1 wins if state is 3, Player 2 loses
            if state == 3:
                return 1 if player_id == 1 else -1
            return 0

        def pretty_str(self, state: int):
            return str(state)

    def test_select_action(self):
        state = 0
        action = self.player1.select_action(state, self.game.legal_actions(state))
        self.assertEqual(action, 2, "Player 1 should choose action 2")

    def test_evaluate_terminal_state(self):
        terminal_state = 3
        score_player1 = self.player1.evaluate(terminal_state)
        score_player2 = self.player2.evaluate(terminal_state)
        self.assertEqual(score_player1, 1, "Player 1 should win in state 3")
        self.assertEqual(score_player2, -1, "Player 2 should lose in state 3")

    def test_minimax_depth_limit_with_heuristic(self):
        # This test now includes non-terminal states evaluated with the heuristic
        state = 2  # Non-terminal state
        shallow_player = MinimaxPlayer(self.game, player_id=1, max_depth=1)
        deep_player = MinimaxPlayer(self.game, player_id=1, max_depth=3)

        shallow_action = shallow_player.select_action(state, self.game.legal_actions(state))
        deep_action = deep_player.select_action(state, self.game.legal_actions(state))

        self.assertEqual(shallow_action, deep_action, "Both should choose the same optimal action with lookahead")


if __name__ == "__main__":
    unittest.main()
