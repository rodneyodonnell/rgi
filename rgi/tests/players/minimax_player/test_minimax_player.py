from typing import Any
import pytest
from rgi.core.base import Game
from rgi.players.minimax_player.minimax_player import MinimaxPlayer
from rgi.games.count21.count21 import Count21Game, Count21State

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def game() -> Game[Any, int]:
    return Count21Game(target=21)


@pytest.fixture
def player1(game: Game[Any, int]) -> MinimaxPlayer[Any, Any]:
    return MinimaxPlayer[Any, Any](game, player_id=1, max_depth=2)


@pytest.fixture
def player2(game: Game[Any, int]) -> MinimaxPlayer[Any, Any]:
    return MinimaxPlayer[Any, Any](game, player_id=2, max_depth=2)


def run_minimax(target: int, max_depth: int) -> int:
    game = Count21Game(target=target)
    state = game.initial_state()
    player = MinimaxPlayer(game, player_id=1, max_depth=max_depth)
    action = player.select_action(state, game.legal_actions(state))
    return action


def test_minimax_depth() -> None:
    # Trivial case, choose one less than target
    assert run_minimax(target=3, max_depth=5) == 2
    assert run_minimax(target=4, max_depth=5) == 3

    # depth 10 search is required to find the optimal move (optimal=3).
    # At lower depths the first option is chosen (first=1)
    assert run_minimax(target=20, max_depth=9) == 1
    assert run_minimax(target=20, max_depth=10) == 3


def test_evaluate_terminal_state(player1: MinimaxPlayer[Any, Any], player2: MinimaxPlayer[Any, Any]) -> None:
    terminal_state = Count21State(score=21, current_player=2)  # Total 21, Player 2 wins
    score_player1 = player1.evaluate(terminal_state)
    score_player2 = player2.evaluate(terminal_state)
    assert score_player1 == -1, "Player 1 should lose in this terminal state"
    assert score_player2 == 1, "Player 2 should win in this terminal state"


def test_minimax_player_perfect_play(game: Game[Any, int]) -> None:
    player1 = MinimaxPlayer(game, player_id=1, max_depth=10)
    player2 = MinimaxPlayer(game, player_id=2, max_depth=10)

    state = game.initial_state()
    current_player = 1
    action_sequence = []

    while not game.is_terminal(state):
        if current_player == 1:
            action = player1.select_action(state, game.legal_actions(state))
        else:
            action = player2.select_action(state, game.legal_actions(state))
        action_sequence.append(action)

        state = game.next_state(state, action)
        current_player = 3 - current_player  # Switch players

    # With perfect play, player 2 should always win
    assert game.reward(state, 2) == 1, "Player 2 should win with perfect play"
    assert game.reward(state, 1) == -1, "Player 1 should lose with perfect play"
    assert action_sequence == [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1]
