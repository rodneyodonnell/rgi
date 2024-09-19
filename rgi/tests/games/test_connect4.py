import pytest
import textwrap
from rgi.games.connect4 import Connect4Game, Connect4State


@pytest.fixture
def game():
    return Connect4Game()


def test_initial_state(game):
    state = game.initial_state()
    assert state.current_player == 1
    assert all(state.board.get((row, col)) is None for row in range(1, 6 + 1) for col in range(1, 7 + 1))


def test_legal_actions(game):
    state = game.initial_state()
    assert game.legal_actions(state) == list(range(1, 7 + 1))

    # Fill up a column
    for _ in range(1, 6 + 1):
        state = game.next_state(state, action=1)
    assert game.legal_actions(state) == list(range(2, 7 + 1))


def test_next_state(game):
    state = game.initial_state()
    next_state = game.next_state(state, action=3)
    assert next_state.current_player == 2
    assert next_state.board.get((1, 3)) == 1


def test_is_terminal(game):
    state = game.initial_state()
    assert not game.is_terminal(state)

    # Create a winning state
    for i in range(1, 3 + 1):
        state = game.next_state(state, action=i)
        assert not game.is_terminal(state)
        state = game.next_state(state, action=i)
        assert not game.is_terminal(state)
    state = game.next_state(state, action=4)
    print(game.pretty_str(state))
    assert game.is_terminal(state)


def test_reward(game):
    state = game.initial_state()
    assert game.reward(state, 1) == 0
    assert game.reward(state, 2) == 0

    # Create a winning state for player 1
    for i in range(4):
        state = game.next_state(state, action=i + 1)
        if i < 3:
            state = game.next_state(state, action=i + 1)
    assert game.reward(state, 1) == 1
    assert game.reward(state, 2) == -1


def test_vertical_win(game):
    state = game.initial_state()
    for _ in range(3):
        state = game.next_state(state, action=1)
        state = game.next_state(state, action=2)
    state = game.next_state(state, action=1)
    assert game.is_terminal(state)
    assert game.reward(state, 1) == 1


def test_horizontal_win(game):
    state = game.initial_state()
    for i in range(4):
        state = game.next_state(state, action=i + 1)
        if i < 3:
            state = game.next_state(state, action=1)
    assert game.is_terminal(state)
    assert game.reward(state, 1) == 1


def test_diagonal_win(game):
    state = game.initial_state()
    moves = [0, 1, 1, 2, 2, 3, 2, 3, 3, 0, 3]
    for move in moves:
        state = game.next_state(state, move + 1)
    assert game.is_terminal(state)
    assert game.reward(state, 1) == 1


def test_invalid_move(game):
    state = game.initial_state()
    with pytest.raises(ValueError):
        game.next_state(state, 8)


def test_custom_board_size():
    game = Connect4Game(width=8, height=7, connect=5)
    state = game.initial_state()
    assert len(game.legal_actions(state)) == 8
    for _ in range(7):
        state = game.next_state(state, 1)
    assert len(game.legal_actions(state)) == 7


@pytest.mark.parametrize("verbose", [True, False])
def test_draw(game, verbose):
    state = game.initial_state()
    # fmt: off
    moves = [
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6, 7,
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6,
        7, 7, 7, 7,
    ]
    # fmt: on
    for i, move in enumerate(moves):
        state = game.next_state(state, move)
        assert not game.is_terminal(state)
        assert game.reward(state, 1) == 0
        assert game.reward(state, 2) == 0
        if verbose:
            print(f"Debug - Move {i+1}: {move}")
            print(game.pretty_str(state))
            print(f"Is terminal: {game.is_terminal(state)}\n")

    state = game.next_state(state, 7)
    assert game.is_terminal(state)
    assert game.reward(state, 1) == 0
    assert game.reward(state, 2) == 0


@pytest.mark.parametrize(
    "board_str, expected_player",
    [
        (
            textwrap.dedent(
                """
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    +-+-+-+-+-+-+-+
    """
            ),
            2,
        ),
        (
            textwrap.dedent(
                """
    | | | | | | | |
    | | | | | | | |
    | | | | | | | |
    | | | | | | |●|
    |○| | | | | |●|
    |○|○| | | | |●|
    +-+-+-+-+-+-+-+
    """
            ),
            2,
        ),
        (
            textwrap.dedent(
                """
    |○| | | | | | |
    |○| | |●| | | |
    |●| | |○| | | |
    |○|○| |●| | | |
    |○|○|●|●| |●| |
    |○|●|○|●|●|●|○|
    +-+-+-+-+-+-+-+
    """
            ),
            2,
        ),
    ],
)
def test_connect4_parse_and_pretty_print(game, board_str, expected_player):
    state = game.parse_board(board_str, current_player=expected_player)
    pretty_printed = game.pretty_str(state)
    assert pretty_printed.strip() == board_str.strip()
