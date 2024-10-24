import pytest
import torch
from rgi.games.count21 import Count21Game, Count21Serializer


@pytest.fixture
def game():
    return Count21Game()


@pytest.fixture
def serializer():
    return Count21Serializer()


def test_initial_state(game):
    assert game.initial_state() == (0,)


def test_current_player_id(game):
    assert game.current_player_id((0,)) == 1
    assert game.current_player_id((0, 1)) == 2
    assert game.current_player_id((0, 1, 2)) == 1


def test_legal_actions(game):
    assert game.legal_actions((0,)) == [1, 2, 3]


def test_next_state(game):
    assert game.next_state((0,), 2) == (0, 2)
    assert game.next_state((0, 2), 3) == (0, 2, 3)


def test_is_terminal(game):
    assert not game.is_terminal((0,))
    assert not game.is_terminal((0, 2, 3, 4, 5))
    assert game.is_terminal((0, 2, 3, 4, 5, 7))


def test_reward(game):
    non_terminal_state = (0, 2, 3, 4, 5)
    assert game.reward(non_terminal_state, 1) == 0.0
    assert game.reward(non_terminal_state, 2) == 0.0

    terminal_state = (0, 2, 3, 4, 5, 7)
    assert game.reward(terminal_state, 1) == -1.0
    assert game.reward(terminal_state, 2) == 1.0


def test_pretty_str(game):
    assert game.pretty_str((0, 2, 3, 4)) == "Count: 9, Moves: (0, 2, 3, 4)"


def test_serializer(game, serializer):
    state = (0, 2, 3, 4)
    serialized_state = serializer.serialize_state(game, state)
    assert serialized_state == {"state": (0, 2, 3, 4)}

    action_data = {"action": 2}
    parsed_action = serializer.parse_action(game, action_data)
    assert parsed_action == 2

    state_tensor = serializer.state_to_tensor(game, state)
    assert torch.equal(state_tensor, torch.tensor([0, 2, 3, 4], dtype=torch.long))

    action_tensor = serializer.action_to_tensor(game, 2)
    assert torch.equal(action_tensor, torch.tensor(2, dtype=torch.long))

    deserialized_action = serializer.tensor_to_action(game, torch.tensor(2))
    assert deserialized_action == 2

    deserialized_state = serializer.tensor_to_state(game, torch.tensor([0, 2, 3, 4]))
    assert deserialized_state == (0, 2, 3, 4)
