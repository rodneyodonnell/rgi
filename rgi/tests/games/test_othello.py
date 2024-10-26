from typing import Literal
import textwrap
import pytest

import torch

from rgi.games import othello
from rgi.games.othello import OthelloGame, OthelloState, OthelloSerializer, Action
from rgi.players.random_player import RandomPlayer

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive

TPlayerId = Literal[1, 2]
TPosition = tuple[int, int]


@pytest.fixture
def game() -> OthelloGame:
    return OthelloGame()


@pytest.fixture
def serializer() -> OthelloSerializer:
    return OthelloSerializer()


def test_initial_state(game: OthelloGame) -> None:
    state = game.initial_state()
    assert state.current_player == 1

    # Check initial discs
    mid = game.board_size // 2
    expected_initial_positions = {
        (mid - 1, mid - 1): 2,
        (mid - 1, mid): 1,
        (mid, mid - 1): 1,
        (mid, mid): 2,
    }
    for pos, player in expected_initial_positions.items():
        assert state.board[pos] == player


def test_legal_actions_initial(game: OthelloGame) -> None:
    state = game.initial_state()
    legal_moves = game.legal_actions(state)
    expected_moves = [(2, 3), (3, 2), (4, 5), (5, 4)]
    assert set(legal_moves) == set(expected_moves)


def test_next_state_simple_move(game: OthelloGame) -> None:
    state = game.initial_state()
    action: TPosition = (2, 3)  # A legal move for Black at the start
    next_state = game.next_state(state, action)

    # Verify current player has switched
    assert next_state.current_player == 2

    # Verify the action has been placed
    assert next_state.board[action] == 1

    # Verify the opponent's disc has been flipped
    flipped_pos: TPosition = (3, 3)
    assert next_state.board[flipped_pos] == 1


def test_illegal_move(game: OthelloGame) -> None:
    state = game.initial_state()
    illegal_action: TPosition = (0, 0)  # An empty corner at the start, which is illegal
    with pytest.raises(ValueError):
        game.next_state(state, illegal_action)


def test_pass_turn(game: OthelloGame) -> None:
    # Create a state where the current player has no legal moves
    board_str = """
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    ● ○ ● . . . . .
    ○ ○ ○ . . . . .
    ● ○ ● . . . . .
    """
    state = game.parse_board(board_str, current_player=1, is_terminal=False)

    legal_moves = game.legal_actions(state)
    assert len(legal_moves) == 0


def test_is_terminal(game: OthelloGame) -> None:
    # Fill the board completely
    board = torch.ones((8, 8), dtype=torch.int8)
    state = OthelloState(board=board, current_player=1, is_terminal=True)
    assert game.is_terminal(state)
    state = OthelloState(board=board, current_player=1, is_terminal=False)
    assert not game.is_terminal(state)


def test_reward_win(game: OthelloGame) -> None:
    # Create a winning state for player 1
    board = torch.ones((8, 8), dtype=torch.int8)
    state = OthelloState(board=board, current_player=1, is_terminal=True)
    assert game.reward(state, 1) == 1.0
    assert game.reward(state, 2) == -1.0


def test_reward_draw(game: OthelloGame) -> None:
    # Create a draw state
    board = torch.ones((8, 8), dtype=torch.int8)
    board[::2, ::2] = 2
    board[1::2, 1::2] = 2
    state = OthelloState(board=board, current_player=1, is_terminal=True)
    assert game.reward(state, 1) == 0.0
    assert game.reward(state, 2) == 0.0


def test_full_game_simulation(game: OthelloGame) -> None:
    # Simulate a short sequence of moves
    state = game.initial_state()
    moves: list[TPosition] = [
        (2, 3),  # Player 1
        (2, 2),  # Player 2
        (2, 1),  # Player 1
        (4, 2),  # Player 2
        (1, 1),  # Player 1
    ]
    for move in moves:
        print(game.pretty_str(state))
        print(f"player: {game.current_player_id(state)} move: {move} legal: {game.legal_actions(state)}")
        state = game.next_state(state, move)

    # Verify board state after moves
    expected_board_str = """
        . . . . . . . .
        ● . . . . . . .
        ● ● ● . . . . .
        . ● ● ● . . . .
        . ○ ○ ○ . . . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        """
    expected_state = game.parse_board(expected_board_str, current_player=2, is_terminal=False)

    assert torch.all(expected_state.board == state.board)


def test_edge_flipping(game: OthelloGame) -> None:
    # Test flipping discs on the edge of the board
    board = torch.zeros((8, 8), dtype=torch.int8)
    board[0, :7] = torch.tensor([2, 1, 1, 1, 1, 1, 1])
    state = OthelloState(board=board, current_player=2, is_terminal=False)

    # Player 2 places at (0, 7), which should flip discs from (0,1)-(0,6)
    action: TPosition = (0, 7)
    next_state = game.next_state(state, action)

    assert torch.all(next_state.board[0, 1:8] == 2)


def test_corner_capture(game: OthelloGame) -> None:
    # Test capturing a corner and flipping appropriately
    board = torch.zeros((8, 8), dtype=torch.int8)
    board[range(7), range(7)] = 1
    board[0, 0] = 2
    state = OthelloState(board=board, current_player=2, is_terminal=False)

    # Player 2 places at (7,7), which should flip discs along the diagonal
    action: TPosition = (7, 7)
    next_state = game.next_state(state, action)
    assert torch.all(next_state.board[range(8), range(8)] == 2)


def test_no_flip_move(game: OthelloGame) -> None:
    # Attempting to make a move that doesn't flip any discs
    state = game.initial_state()
    illegal_action: TPosition = (0, 0)  # An empty corner at the start, which doesn't flip any discs
    with pytest.raises(ValueError):
        game.next_state(state, illegal_action)


def test_full_board_playthrough(game: OthelloGame) -> None:
    # Simulate a full game with random moves
    state = game.initial_state()
    player1: RandomPlayer[OthelloState, TAction] = RandomPlayer()
    player2: RandomPlayer[OthelloState, TAction] = RandomPlayer()
    players: dict[TPlayerId, RandomPlayer[OthelloState, TAction]] = {
        1: player1,
        2: player2,
    }

    while not game.is_terminal(state):
        current_player_id = game.current_player_id(state)
        legal_actions = game.legal_actions(state)
        action: TAction = players[current_player_id].select_action(state, legal_actions)
        state = game.next_state(state, action)

    # At the end, check that the board is full or no moves are possible
    assert game.is_terminal(state)


def test_flipping_multiple_directions(game: OthelloGame) -> None:
    # Test a move that flips discs in multiple directions
    board_str = """
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . ● ● . . .
    . . ○ ○ ● ● . .
    . . . . ○ ● . .
    . . . . ○ . . .
    ○ . . . . . . .
    """
    state = game.parse_board(board_str, current_player=1, is_terminal=False)

    # Player 1 places at (2,3), which should flip discs in multiple directions
    action: TPosition = (2, 3)
    next_state = game.next_state(state, action)

    expected_board_str = textwrap.dedent(
        """
         . . . . . . . .
         . . . . . . . .
         . . . ● . . . .
         . . . ● ● . . .
         . . ○ ● ● ● . .
         . . . ● ● ● . .
         . . . . ○ . . .
         ○ . . . . . . .
        """
    )
    assert game.pretty_str(next_state).strip() == expected_board_str.strip()


def test_no_legal_moves_for_current_player(game: OthelloGame) -> None:
    # Create a state where the current player has no legal moves but the game is not over
    # Manually set up the board so player 1 has no moves
    board = torch.zeros((8, 8), dtype=torch.int8)
    board[0:2, 0:2] = torch.tensor([[2, 2], [2, 1]])
    state = OthelloState(board=board, current_player=1, is_terminal=False)

    legal_moves = game.legal_actions(state)
    assert len(legal_moves) == 0
    assert not game.is_terminal(state)


def test_game_end_by_no_moves(game: OthelloGame) -> None:
    board_str = """
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    ● ○ ○ . . . . .
    """
    state = game.parse_board(board_str, current_player=1, is_terminal=False)
    new_state = game.next_state(state, (0, 3))
    assert game.is_terminal(new_state)


@pytest.mark.parametrize(
    "board_str, positions",
    [
        (
            """
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . ○ . . . . . .
    """,
            {(7, 1): 2},
        ),
        (
            """
    . . . . . . . .
    . . . . . . . .
    . . . ○ ● . . .
    . . ○ ● ○ ● . .
    . . ● ○ ● ○ . .
    . . . ● ○ . . .
    . . . . . . . .
    . . . . . . . .
    """,
            {
                (2, 3): 1,
                (2, 4): 2,
                (3, 2): 1,
                (3, 3): 2,
                (3, 4): 1,
                (3, 5): 2,
                (4, 2): 2,
                (4, 3): 1,
                (4, 4): 2,
                (4, 5): 1,
                (5, 3): 2,
                (5, 4): 1,
            },
        ),
    ],
)
def test_parse_board(game: OthelloGame, board_str: str, positions: dict[TPosition, TPlayerId]) -> None:
    state = game.parse_board(board_str, current_player=1, is_terminal=False)
    # Verify the board is parsed correctly
    for pos, player in positions.items():
        assert state.board[pos] == player


def test_pretty_str_bottom_left(game: OthelloGame) -> None:
    # Define a state with pieces at (7,0) and (6,0)
    board = torch.zeros((8, 8), dtype=torch.int8)
    board[7, 0] = 2
    board[6, 0] = 2
    state = OthelloState(board=board, current_player=1, is_terminal=False)
    expected_output = textwrap.dedent(
        """
         . . . . . . . .
         . . . . . . . .
         . . . . . . . .
         . . . . . . . .
         . . . . . . . .
         . . . . . . . .
         ○ . . . . . . .
         ○ . . . . . . .
        """
    )
    assert game.pretty_str(state).strip() == expected_output.strip()


def test_state_to_tensor(game: OthelloGame, serializer: OthelloSerializer):
    state = game.initial_state()
    tensor = serializer.state_to_tensor(game, state)
    assert tensor.shape == torch.Size([65])  # 8x8 board + 1 for current player
    assert tensor[8 * 3 + 3] == 2  # Check (3,3) position
    assert tensor[8 * 3 + 4] == 1  # Check (3,4) position
    assert tensor[-1] == 1  # Check current player


def test_action_to_tensor(game: OthelloGame, serializer: OthelloSerializer):
    action = (3, 4)
    tensor = serializer.action_to_tensor(game, action)
    assert torch.equal(tensor, torch.tensor([3, 4]))


def test_tensor_to_action(game: OthelloGame, serializer: OthelloSerializer):
    tensor = torch.tensor([3, 4])
    action = serializer.tensor_to_action(game, tensor)
    assert action == (3, 4)


def test_tensor_to_state(game: OthelloGame, serializer: OthelloSerializer):
    tensor = torch.zeros(65)
    tensor[8 * 3 + 3] = 2
    tensor[8 * 3 + 4] = 1
    tensor[8 * 4 + 3] = 1
    tensor[8 * 4 + 4] = 2
    tensor[-1] = 1  # Set current player to 1
    state = serializer.tensor_to_state(game, tensor)
    assert state.board[3, 3] == 2
    assert state.board[3, 4] == 1
    assert state.board[4, 3] == 1
    assert state.board[4, 4] == 2
    assert state.current_player == 1
