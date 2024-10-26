import pytest
import torch
from rgi.games import count21
from rgi.games.count21 import Count21Game, Count21Serializer

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def game() -> Count21Game:
    return Count21Game()


@pytest.fixture
def serializer() -> Count21Serializer:
    return Count21Serializer()


def test_initial_state(game: Count21Game) -> None:
    assert game.initial_state() == count21.GameState(0, 1)


def test_current_player_id(game: Count21Game) -> None:
    assert game.current_player_id(count21.GameState(0, 1)) == 1
    assert game.current_player_id(count21.GameState(0, 2)) == 2


def test_legal_actions(game: Count21Game) -> None:
    assert game.legal_actions(count21.GameState(0, 1)) == (1, 2, 3)


def test_next_state(game: Count21Game) -> None:
    assert game.next_state(count21.GameState(0, 1), 3) == count21.GameState(3, 2)
    assert game.next_state(count21.GameState(3, 2), 3) == count21.GameState(6, 1)
    assert game.next_state(count21.GameState(6, 1), 1) == count21.GameState(7, 2)


def test_is_terminal(game: Count21Game) -> None:
    assert not game.is_terminal(count21.GameState(0, 1))
    assert not game.is_terminal(count21.GameState(20, 1))
    assert game.is_terminal(count21.GameState(21, 1))


def test_reward(game: Count21Game) -> None:

    # non-terminal states
    assert game.reward(count21.GameState(20, 1), 1) == 0.0
    assert game.reward(count21.GameState(20, 1), 2) == 0.0

    # terminal state
    assert game.reward(count21.GameState(21, 1), 1) == 1.0
    assert game.reward(count21.GameState(21, 1), 2) == -1.0


def test_pretty_str(game: Count21Game) -> None:
    assert game.pretty_str(count21.GameState(9, 1)) == "Score: 9, Player: 1"


def test_serializer(game: Count21Game, serializer: Count21Serializer) -> None:
    game_state = count21.GameState(9, 1)
    serialized_state = serializer.serialize_state(game, game_state)
    assert serialized_state == {"score": 9, "current_player": 1}

    action_data = {"action": 2}
    parsed_action = serializer.parse_action(game, action_data)
    assert parsed_action == 2


def test_batch_state(game: Count21Game) -> None:
    game_state = game.initial_state()
    game_state_list = [game_state]
    game_action_list = []
    for action in [1, 3, 2, 1]:
        game_state = game.next_state(game_state, action)
        game_state_list.append(game_state)
        game_action_list.append(action)

    batch_game_state = count21.BatchGameState.from_sequence(game_state_list)
    assert torch.equal(batch_game_state.score, torch.tensor([0, 1, 4, 6, 7]))
    assert torch.equal(batch_game_state.current_player, torch.tensor([1, 2, 1, 2, 1]))
    assert list(batch_game_state) == game_state_list

    batch_game_action = count21.BatchAction.from_sequence(game_action_list)
    assert torch.equal(batch_game_action.values, torch.tensor([1, 3, 2, 1]))
    assert list(batch_game_action) == game_action_list
