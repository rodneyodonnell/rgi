from typing import Literal
import textwrap
import pytest

from immutables import Map

from rgi.games.othello import OthelloGame, OthelloState, TAction
from rgi.players.random_player import RandomPlayer

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive

TPlayerId = Literal[1, 2]
TPosition = tuple[int, int]


@pytest.fixture
def game() -> OthelloGame:
    return OthelloGame()


def test_initial_state(game: OthelloGame) -> None:
    state = game.initial_state()
    assert state.current_player == 1

    # Check initial discs
    mid = game.board_size // 2
    expected_initial_positions: dict[TPosition, TPlayerId] = {
        (mid, mid): 2,
        (mid + 1, mid + 1): 2,
        (mid, mid + 1): 1,
        (mid + 1, mid): 1,
    }
    for pos, player in expected_initial_positions.items():
        assert state.board.get(pos) == player


def test_legal_actions_initial(game: OthelloGame) -> None:
    state = game.initial_state()
    legal_moves = game.legal_actions(state)
    expected_moves: list[TPosition] = [
        (3, 4),  # Up from (4,4)
        (4, 3),  # Left from (4,4)
        (5, 6),  # Right from (5,5)
        (6, 5),  # Down from (5,5)
    ]
    assert set(legal_moves) == set(expected_moves)


def test_next_state_simple_move(game: OthelloGame) -> None:
    state = game.initial_state()
    action: TPosition = (3, 4)  # A legal move for Black at the start
    next_state = game.next_state(state, action)

    # Verify current player has switched
    assert next_state.current_player == 2

    # Verify the action has been placed
    assert next_state.board.get(action) == 1

    # Verify the opponent's disc has been flipped
    flipped_pos: TPosition = (4, 4)
    assert next_state.board.get(flipped_pos) == 1


def test_illegal_move(game: OthelloGame) -> None:
    state = game.initial_state()
    illegal_action: TPosition = (1, 1)  # An empty corner at the start, which is illegal
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
    board: Map[TPosition, int] = Map()
    state = OthelloState(board=board, current_player=1, is_terminal=True)
    assert game.is_terminal(state)
    state = OthelloState(board=board, current_player=1, is_terminal=False)
    assert not game.is_terminal(state)


def test_reward_win(game: OthelloGame) -> None:
    # Create a winning state for player 1
    board: Map[TPosition, int] = Map(
        {(row, col): 1 for row in range(1, game.board_size + 1) for col in range(1, game.board_size + 1)}
    )
    state = OthelloState(board=board, current_player=1, is_terminal=True)
    assert game.reward(state, 1) == 1.0
    assert game.reward(state, 2) == -1.0


def test_reward_draw(game: OthelloGame) -> None:
    # Create a draw state
    board: Map[TPosition, int] = Map(
        {
            (row, col): 1 if (row + col) % 2 == 0 else 2
            for row in range(1, game.board_size + 1)
            for col in range(1, game.board_size + 1)
        }
    )
    state = OthelloState(board=board, current_player=1, is_terminal=True)
    assert game.reward(state, 1) == 0.0
    assert game.reward(state, 2) == 0.0


def test_full_game_simulation(game: OthelloGame) -> None:
    # Simulate a short sequence of moves
    state = game.initial_state()
    moves: list[TPosition] = [
        (4, 3),  # Player 1
        (3, 3),  # Player 2
        (3, 4),  # Player 1
        (5, 3),  # Player 2
        (5, 2),  # Player 1
    ]
    for move in moves:
        print(game.pretty_str(state))
        print(f"player: {game.current_player_id(state)} move: {move} legal: {game.legal_actions(state)}")
        state = game.next_state(state, move)

    # Verify board state after moves
    expected_board_str = """
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        . ● ○ ○ ○ . . .
        . . ● ● ● . . .
        . . ○ ● . . . .
        . . . . . . . .
        . . . . . . . .
        """
    expected_state = game.parse_board(expected_board_str, current_player=1, is_terminal=False)

    assert expected_state.board == state.board


def test_edge_flipping(game: OthelloGame) -> None:
    # Test flipping discs on the edge of the board
    board: Map[TPosition, int] = Map(
        {
            (1, 1): 2,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 1,
            (1, 5): 1,
            (1, 6): 1,
            (1, 7): 1,
        }
    )
    state = OthelloState(board=board, current_player=2, is_terminal=False)

    # Player 2 places at (1, 8), which should flip discs from (1,2)-(1,7)
    action: TPosition = (1, 8)
    next_state = game.next_state(state, action)

    for col in range(2, 8):
        assert next_state.board.get((1, col)) == 2


def test_corner_capture(game: OthelloGame) -> None:
    # Test capturing a corner and flipping appropriately
    board: Map[TPosition, int] = Map(
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
    state = OthelloState(board=board, current_player=2, is_terminal=False)

    # Player 2 places at (8,8), which should flip discs along the diagonal
    action: TPosition = (8, 8)
    next_state = game.next_state(state, action)
    for i in range(2, 8):
        assert next_state.board.get((i, i)) == 2


def test_no_flip_move(game: OthelloGame) -> None:
    # Attempting to make a move that doesn't flip any discs
    state = game.initial_state()
    illegal_action: TPosition = (
        1,
        1,
    )  # An empty corner at the start, which doesn't flip any discs
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

    # Player 1 places at (3,4), which should flip discs in multiple directions
    action: TPosition = (3, 4)
    next_state = game.next_state(state, action)

    expected_board_str = textwrap.dedent(
        """
         . . . . . . . .
         . . . . . . . .
         . . . . . . . .
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
    board: Map[TPosition, int] = Map(
        {
            (1, 1): 2,
            (1, 2): 2,
            (2, 1): 2,
            (2, 2): 1,
        }
    )
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
    new_state = game.next_state(state, (1, 4))
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
            {(1, 2): 2},
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
            },
        ),
    ],
)
def test_parse_board(game: OthelloGame, board_str: str, positions: dict[TPosition, TPlayerId]) -> None:
    state = game.parse_board(board_str, current_player=1, is_terminal=False)
    # Verify the board is parsed correctly
    for pos, player in positions.items():
        assert state.board.get(pos) == player


def test_pretty_str_bottom_left(game: OthelloGame) -> None:
    # Define a state with pieces at (1,1) and (2,1)
    state = OthelloState(board=Map({(1, 1): 2, (2, 1): 2}), current_player=1, is_terminal=False)
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
