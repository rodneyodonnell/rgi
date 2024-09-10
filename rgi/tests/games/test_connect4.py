import unittest
from rgi.games.connect4 import Connect4Game, Connect4State


class TestConnect4Game(unittest.TestCase):
    def setUp(self):
        self.game = Connect4Game()

    def test_initial_state(self):
        state = self.game.initial_state()
        self.assertEqual(state.current_player, 1)
        self.assertTrue(all(state.board.get((row, col)) == 0 for row in range(6) for col in range(7)))

    def test_legal_actions(self):
        state = self.game.initial_state()
        self.assertEqual(self.game.legal_actions(state), list(range(7)))

        # Fill up a column
        for _ in range(6):
            state = self.game.next_state(state, 0)
        self.assertEqual(self.game.legal_actions(state), list(range(1, 7)))

    def test_next_state(self):
        state = self.game.initial_state()
        next_state = self.game.next_state(state, 3)
        self.assertEqual(next_state.current_player, 2)
        self.assertEqual(next_state.board.get((5, 3)), 1)

    def test_is_terminal(self):
        state = self.game.initial_state()
        self.assertFalse(self.game.is_terminal(state))

        # Create a winning state
        for i in range(4):
            state = self.game.next_state(state, i)
            state = self.game.next_state(state, i)
        self.assertTrue(self.game.is_terminal(state))

    def test_reward(self):
        state = self.game.initial_state()
        self.assertEqual(self.game.reward(state, 1), 0)
        self.assertEqual(self.game.reward(state, 2), 0)

        # Create a winning state for player 1
        for i in range(4):
            state = self.game.next_state(state, i)
            if i < 3:
                state = self.game.next_state(state, i)
        self.assertEqual(self.game.reward(state, 1), 1)
        self.assertEqual(self.game.reward(state, 2), -1)

    def test_vertical_win(self):
        state = self.game.initial_state()
        for _ in range(3):
            state = self.game.next_state(state, 0)
            state = self.game.next_state(state, 1)
        state = self.game.next_state(state, 0)
        self.assertTrue(self.game.is_terminal(state))
        self.assertEqual(self.game.reward(state, 1), 1)

    def test_horizontal_win(self):
        state = self.game.initial_state()
        for i in range(4):
            state = self.game.next_state(state, i)
            if i < 3:
                state = self.game.next_state(state, 0)
        self.assertTrue(self.game.is_terminal(state))
        self.assertEqual(self.game.reward(state, 1), 1)

    def test_diagonal_win(self):
        state = self.game.initial_state()
        moves = [0, 1, 1, 2, 2, 3, 2, 3, 3, 0, 3]
        for move in moves:
            state = self.game.next_state(state, move)
        self.assertTrue(self.game.is_terminal(state))
        self.assertEqual(self.game.reward(state, 1), 1)

    def test_invalid_move(self):
        state = self.game.initial_state()
        with self.assertRaises(ValueError):
            self.game.next_state(state, 7)

    def test_custom_board_size(self):
        game = Connect4Game(width=8, height=7, connect=5)
        state = game.initial_state()
        self.assertEqual(len(game.legal_actions(state)), 8)
        for _ in range(7):
            state = game.next_state(state, 0)
        self.assertEqual(len(game.legal_actions(state)), 7)

    def test_draw(self, verbose=True):
        # Create a full board with no winner
        state = self.game.initial_state()
        # fmt: off
        moves = [
            0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5, 6,
            0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5,
            6, 6, 6, 6,
        ]
        # fmt: on
        for i, move in enumerate(moves):
            state = self.game.next_state(state, move)
            self.assertFalse(self.game.is_terminal(state))
            self.assertEqual(self.game.reward(state, 1), 0)
            self.assertEqual(self.game.reward(state, 2), 0)
            self.assertEqual(self.game._check_winner(state), None)
            if verbose:
                print(f"Debug - Move {i+1}: {move}")
                print(self.game.pretty_str(state))
                print(f"Is terminal: {self.game.is_terminal(state)}\n")

        state = self.game.next_state(state, 6)
        self.assertTrue(self.game.is_terminal(state))
        self.assertEqual(self.game.reward(state, 1), 0)
        self.assertEqual(self.game.reward(state, 2), 0)
        self.assertEqual(self.game._check_winner(state), None)


if __name__ == "__main__":
    unittest.main()
