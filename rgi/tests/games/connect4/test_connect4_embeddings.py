import pytest

import torch

from rgi.games.connect4.connect4_embeddings import Connect4StateEmbedder, Connect4ActionEmbedder
from rgi.games.connect4 import Connect4Game, Connect4Serializer
from rgi.games import connect4

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def game() -> Connect4Game:
    return Connect4Game()


@pytest.fixture
def state_embedder() -> Connect4StateEmbedder:
    state_embedder = Connect4StateEmbedder()
    state_embedder.eval()  # Set to evaluation mode
    return state_embedder


@pytest.fixture
def action_embedder() -> Connect4ActionEmbedder:
    action_embedder = Connect4ActionEmbedder()
    action_embedder.eval()  # Set to evaluation mode
    return action_embedder


def test_connect4_state_embedder(game: Connect4Game, state_embedder: Connect4StateEmbedder) -> None:
    init_state = game.initial_state()
    state2 = game.next_state(init_state, 1)
    batch_state = connect4.BatchGameState.from_sequence([init_state, state2])

    embedding = state_embedder(batch_state)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (2, 64)  # Now we have a batch dimension


def test_connect4_action_embedder(action_embedder: Connect4ActionEmbedder) -> None:
    # Create a batch of actions
    batch_action = connect4.BatchAction.from_sequence([1, 2, 3, 4, 5, 6, 7])
    embeddings = action_embedder(batch_action)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (7, 64)  # 7 actions, each embedded to 64 dimensions


def test_invalid_inputs(state_embedder: Connect4StateEmbedder, action_embedder: Connect4ActionEmbedder) -> None:
    with pytest.raises((TypeError, AttributeError)):
        state_embedder("not a tensor")

    with pytest.raises(IndexError):  # or ValueError, depending on how you implement error checking
        action_embedder(connect4.BatchAction.from_sequence([0]))

    with pytest.raises(IndexError):  # or ValueError, depending on how you implement error checking
        action_embedder(action_embedder(connect4.BatchAction.from_sequence([8])))


def test_deterministic_output(
    game: Connect4Game, state_embedder: Connect4StateEmbedder, action_embedder: Connect4ActionEmbedder
) -> None:
    init_state = game.initial_state()
    state2 = game.next_state(init_state, 1)
    batch_state = connect4.BatchGameState.from_sequence([init_state, state2])

    batch_action = connect4.BatchAction.from_sequence([1, 2])

    # Run embeddings twice and check if they're the same
    state_emb1 = state_embedder(batch_state)
    state_emb2 = state_embedder(batch_state)
    assert torch.allclose(state_emb1, state_emb2)

    action_emb1 = action_embedder(batch_action)
    action_emb2 = action_embedder(batch_action)
    assert torch.allclose(action_emb1, action_emb2)
