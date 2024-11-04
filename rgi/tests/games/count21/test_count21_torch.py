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
