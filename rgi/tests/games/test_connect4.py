import unittest
from rgi.games.connect4 import Connect4


class TestConnect4(unittest.TestCase):
    def setUp(self):
        self.game = Connect4()

    def test_initial_state(self):
        state = self.game.initial_state()
        self.assertEqual(len(state), 6)
        self.assertEqual(len(state[0]), 7)
        self.assertTrue(all(cell == 0 for row in state for cell in row))

    def test_current_player(self):
        state = self.game.initial_state()
        self.assertEqual(self.game.current_player(state), 1)
        state[5][0] = 1
        self.assertEqual(self.game.current_player(state), 2)

    def test_legal_actions(self):
        state = self.game.initial_state()
        self.assertEqual(self.game.legal_actions(state), list(range(7)))
        
        # Fill a column
        for i in range(6):
            state[i][0] = 1
        self.assertEqual(self.game.legal_actions(state), list(range(1, 7)))

    def test_next_state(self):
        state = self.game.initial_state()
        new_state = self.game.next_state(state, 3)
        self.assertEqual(new_state[5][3], 1)
        self.assertEqual(self.game.current_player(new_state), 2)

    def test_is_terminal(self):
        state = self.game.initial_state()
        self.assertFalse(self.game.is_terminal(state))
        
        # Horizontal win
        for i in range(4):
            state[5][i] = 1
        self.assertTrue(self.game.is_terminal(state))

    def test_reward(self):
        state = self.game.initial_state()
        self.assertEqual(self.game.reward(state, 1), 0.0)
        
        # Player 1 wins
        for i in range(4):
            state[5][i] = 1
        self.assertEqual(self.game.reward(state, 1), 1.0)
        self.assertEqual(self.game.reward(state, 2), -1.0)

    def test_horizontal_win(self):
        state = self.game.initial_state()
        for i in range(4):
            state[0][i] = 1
        self.assertTrue(self.game._check_win(state, 1))

    def test_vertical_win(self):
        state = self.game.initial_state()
        for i in range(4):
            state[i][0] = 2
        self.assertTrue(self.game._check_win(state, 2))

    def test_diagonal_win(self):
        state = self.game.initial_state()
        for i in range(4):
            state[i][i] = 1
        self.assertTrue(self.game._check_win(state, 1))

    def test_reverse_diagonal_win(self):
        state = self.game.initial_state()
        for i in range(4):
            state[i][3-i] = 2
        self.assertTrue(self.game._check_win(state, 2))

if __name__ == '__main__':
    unittest.main()