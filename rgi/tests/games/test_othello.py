# rgi/tests/games/test_othello.py

import unittest
from rgi.games.othello import OthelloGame, OthelloState
from immutables import Map


class TestOthelloGame(unittest.TestCase):
    def setUp(self):
        self.game = OthelloGame()
        self.board_size = self.game.board_size

    def test_initial_state(self):
        state = self.game.initial_state()
        self.assertEqual(state.current_player, 1)

        # Check initial discs
        mid = self.board_size // 2
        expected_initial_positions = {
            (mid, mid): 2,
            (mid + 1, mid + 1): 2,
            (mid, mid + 1): 1,
            (mid + 1, mid): 1,
        }
        for pos, player in expected_initial_positions.items():
            self.assertEqual(state.board.get(pos), player)

    def test_legal_actions_initial(self):
        state = self.game.initial_state()
        legal_moves = self.game.legal_actions(state)
        expected_moves = [
            (3, 4),  # Up from (4,4)
            (4, 3),  # Left from (4,4)
            (5, 6),  # Right from (5,5)
            (6, 5),  # Down from (5,5)
        ]
        self.assertEqual(set(legal_moves), set(expected_moves))

    def test_next_state_simple_move(self):
        state = self.game.initial_state()
        action = (3, 4)  # A legal move for Black at the start
        next_state = self.game.next_state(state, action)

        # Verify current player has switched
        self.assertEqual(next_state.current_player, 2)

        # Verify the action has been placed
        self.assertEqual(next_state.board.get(action), 1)

        # Verify the opponent's disc has been flipped
        flipped_pos = (4, 4)
        self.assertEqual(next_state.board.get(flipped_pos), 1)

    def test_illegal_move(self):
        state = self.game.initial_state()
        illegal_action = (1, 1)  # An empty corner at the start, which is illegal
        with self.assertRaises(ValueError):
            self.game.next_state(state, illegal_action)

    def test_pass_turn(self):
        # Create a state where the current player has no legal moves
        state = self.game.initial_state()

        # Manually set up the board
        board = Map(
            {
                (1, 1): 1,
                (1, 2): 2,
                (1, 3): 1,
                (2, 1): 2,
                (2, 2): 1,
                (2, 3): 2,
                (3, 1): 1,
                (3, 2): 2,
                (3, 3): 1,
            }
        )
        state = OthelloState(board=board, current_player=1)

        legal_moves = self.game.legal_actions(state)
        self.assertEqual(len(legal_moves), 0)

        # Next state should keep the same player if the opponent also has no moves
        next_state = self.game.next_state(state, action=None)
        self.assertEqual(next_state.current_player, 2)

    def test_is_terminal(self):
        # Fill the board completely
        board = Map(
            {
                (row, col): 1 if (row + col) % 2 == 0 else 2
                for row in range(1, self.board_size + 1)
                for col in range(1, self.board_size + 1)
            }
        )
        state = OthelloState(board=board, current_player=1)
        self.assertTrue(self.game.is_terminal(state))

    def test_reward_win(self):
        # Create a winning state for player 1
        board = Map({(row, col): 1 for row in range(1, self.board_size + 1) for col in range(1, self.board_size + 1)})
        state = OthelloState(board=board, current_player=1)
        self.assertEqual(self.game.reward(state, 1), 1.0)
        self.assertEqual(self.game.reward(state, 2), -1.0)

    def test_reward_draw(self):
        # Create a draw state
        board = Map(
            {
                (row, col): 1 if (row + col) % 2 == 0 else 2
                for row in range(1, self.board_size + 1)
                for col in range(1, self.board_size + 1)
            }
        )
        state = OthelloState(board=board, current_player=1)
        self.assertEqual(self.game.reward(state, 1), 0.0)
        self.assertEqual(self.game.reward(state, 2), 0.0)

    def test_full_game_simulation(self):
        # Simulate a short sequence of moves
        state = self.game.initial_state()
        moves = [
            (4, 3),  # Player 1
            (3, 3),  # Player 2
            (3, 4),  # Player 1
            (5, 3),  # Player 2
            (5, 4),  # Player 1
        ]
        for move in moves:
            state = self.game.next_state(state, move)

        # Verify board state after moves
        expected_board_positions = {
            (4, 4): 1,
            (5, 5): 1,
            (3, 4): 1,
            (5, 4): 1,
            (4, 3): 1,
            (4, 5): 2,
            (3, 3): 2,
            (5, 3): 2,
        }
        for pos, player in expected_board_positions.items():
            self.assertEqual(state.board.get(pos), player)

    def test_no_legal_moves_for_both_players(self):
        # Create a state where neither player has legal moves
        board = Map()
        state = OthelloState(board=board, current_player=1)
        self.assertTrue(self.game.is_terminal(state))

    def test_pretty_str(self):
        state = self.game.initial_state()
        board_str = self.game.pretty_str(state)
        print("Initial Board:")
        print(board_str)
        # Manually verify the output or write assertions based on expected string

    def test_parse_board(self):
        board_str = """
        . . . . . . . .
        . . . . . . . .
        . . . ● ○ . . .
        . . ● ○ ● ○ . .
        . . ○ ● ○ ● . .
        . . . ○ ● . . .
        . . . . . . . .
        . . . . . . . .
        """
        state = self.game.parse_board(board_str, current_player=1)
        # Verify the board is parsed correctly
        positions = {
            (3, 4): 1,
            (3, 5): 2,
            (4, 3): 1,
            (4, 4): 2,
            (4, 5): 1,
            (4, 6): 2,
            (5, 3): 2,
            (5, 4): 1,
            (5, 5): 2,
            (5, 6): 1,
            (6, 4): 2,
            (6, 5): 1,
        }
        for pos, player in positions.items():
            self.assertEqual(state.board.get(pos), player)

    def test_edge_flipping(self):
        # Test flipping discs on the edge of the board
        state = self.game.initial_state()

        # Set up a specific board state
        board = Map(
            {
                (1, 1): 2,
                (1, 2): 1,
                (1, 3): 1,
                (1, 4): 1,
                (1, 5): 1,
                (1, 6): 1,
                (1, 7): 1,
                (1, 8): 2,
            }
        )
        state = OthelloState(board=board, current_player=2)

        # Player 2 places at (1, 8), which should flip discs from (1,2)-(1,7)
        action = (1, 8)
        next_state = self.game.next_state(state, action)
        for col in range(2, 8):
            self.assertEqual(next_state.board.get((1, col)), 2)

    def test_corner_capture(self):
        # Test capturing a corner and flipping appropriately
        state = self.game.initial_state()

        # Set up a specific board state
        board = Map(
            {
                (1, 1): 2,
                (2, 2): 1,
                (3, 3): 1,
                (4, 4): 1,
                (5, 5): 1,
                (6, 6): 1,
                (7, 7): 1,
            }
        )
        state = OthelloState(board=board, current_player=2)

        # Player 2 places at (8,8), which should flip discs along the diagonal
        action = (8, 8)
        next_state = self.game.next_state(state, action)
        for i in range(2, 8):
            self.assertEqual(next_state.board.get((i, i)), 2)

    def test_no_flip_move(self):
        # Attempting to make a move that doesn't flip any discs
        state = self.game.initial_state()
        illegal_action = (1, 1)  # An empty corner at the start, which doesn't flip any discs
        with self.assertRaises(ValueError):
            self.game.next_state(state, illegal_action)

    def test_full_board_playthrough(self):
        # Simulate a full game with random moves
        from rgi.players.random_player import RandomPlayer

        state = self.game.initial_state()
        player1 = RandomPlayer()
        player2 = RandomPlayer()
        players = {1: player1, 2: player2}

        while not self.game.is_terminal(state):
            current_player_id = self.game.current_player_id(state)
            legal_actions = self.game.legal_actions(state)
            if legal_actions:
                action = players[current_player_id].select_action(state, legal_actions)
                state = self.game.next_state(state, action)
            else:
                # No legal moves, pass turn
                state = OthelloState(state.board, 3 - current_player_id)

        # At the end, check that the board is full or no moves are possible
        self.assertTrue(self.game.is_terminal(state))

    def test_flipping_multiple_directions(self):
        # Test a move that flips discs in multiple directions
        state = self.game.initial_state()

        # Manually set up the board
        board = Map(
            {
                (4, 5): 2,
                (5, 4): 2,
                (5, 5): 1,
                (5, 6): 1,
                (6, 5): 1,
            }
        )
        state = OthelloState(board=board, current_player=1)

        # Player 1 places at (5,5), which should flip discs in multiple directions
        action = (5, 5)
        next_state = self.game.next_state(state, action)

        # Verify that discs are flipped correctly
        flipped_positions = [(4, 5), (5, 4), (5, 6), (6, 5)]
        for pos in flipped_positions:
            self.assertEqual(next_state.board.get(pos), 1)

    def test_no_legal_moves_for_current_player(self):
        # Create a state where the current player has no legal moves but the game is not over
        state = self.game.initial_state()

        # Manually set up the board so player 1 has no moves
        board = Map(
            {
                (1, 1): 2,
                (1, 2): 2,
                (2, 1): 2,
                (2, 2): 1,
            }
        )
        state = OthelloState(board=board, current_player=1)

        legal_moves = self.game.legal_actions(state)
        self.assertEqual(len(legal_moves), 0)
        self.assertFalse(self.game.is_terminal(state))

    def test_game_end_by_no_moves(self):
        # Create a state where neither player has legal moves
        state = self.game.initial_state()
        # Manually set up the board
        board = Map(
            {
                (1, 1): 1,
                (1, 2): 1,
                (2, 1): 1,
                (2, 2): 1,
            }
        )
        state = OthelloState(board=board, current_player=1)
        self.assertTrue(self.game.is_terminal(state))


if __name__ == "__main__":
    unittest.main()
