import pytest
from rgi.core.base import Game
from rgi.players.minimax_player import MinimaxPlayer


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


@pytest.fixture
def game():
    return MockGame()


@pytest.fixture
def player1(game):
    return MinimaxPlayer(game, player_id=1, max_depth=2)


@pytest.fixture
def player2(game):
    return MinimaxPlayer(game, player_id=2, max_depth=2)


def test_select_action(game, player1):
    state = 0
    action = player1.select_action(state, game.legal_actions(state))
    assert action == 2, "Player 1 should choose action 2"


def test_evaluate_terminal_state(player1, player2):
    terminal_state = 3
    score_player1 = player1.evaluate(terminal_state)
    score_player2 = player2.evaluate(terminal_state)
    assert score_player1 == 1, "Player 1 should win in state 3"
    assert score_player2 == -1, "Player 2 should lose in state 3"


@pytest.mark.parametrize(
    "max_depth, expected_action",
    [
        (1, 1),
        (3, 1),
    ],
)
def test_minimax_depth_limit_with_heuristic(game, max_depth, expected_action):
    state = 2  # Non-terminal state
    player = MinimaxPlayer(game, player_id=1, max_depth=max_depth)
    action = player.select_action(state, game.legal_actions(state))
    assert action == expected_action, f"Player with depth {max_depth} should choose action {expected_action}"
