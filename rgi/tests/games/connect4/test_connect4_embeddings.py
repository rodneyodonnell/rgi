from typing import Any
import torch
import pytest
from rgi.games.connect4.connect4_embeddings import Connect4StateEmbedder, Connect4ActionEmbedder
from rgi.games.connect4 import Connect4Game, Connect4Serializer

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive

TParams = dict[str, Any]


@pytest.fixture
def game() -> Connect4Game:
    return Connect4Game()


@pytest.fixture
def serializer() -> Connect4Serializer:
    return Connect4Serializer()


def test_connect4_state_embedder(game: Connect4Game, serializer: Connect4Serializer) -> None:
    init_state = game.initial_state()

    state_embedder = Connect4StateEmbedder()
    state_embedder.eval()  # Set to evaluation mode

    state = game.next_state(init_state, 1)
    encoded_state = serializer.state_to_tensor(game, state)
    # Add batch dimension
    encoded_state_batch = encoded_state.unsqueeze(0)
    embedding = state_embedder(encoded_state_batch)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (1, 64)  # Now we have a batch dimension


def test_connect4_action_embedder() -> None:
    action_embedder = Connect4ActionEmbedder()
    action_embedder.eval()  # Set to evaluation mode

    # Create a batch of actions
    actions = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    embeddings = action_embedder(actions)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (7, 64)  # 7 actions, each embedded to 64 dimensions


def test_invalid_inputs(game: Connect4Game, serializer: Connect4Serializer) -> None:
    state_embedder = Connect4StateEmbedder()
    state_embedder.eval()  # Set to evaluation mode

    action_embedder = Connect4ActionEmbedder()
    action_embedder.eval()  # Set to evaluation mode

    with pytest.raises(TypeError):
        state_embedder("not a tensor")

    with pytest.raises(IndexError):  # or ValueError, depending on how you implement error checking
        action_embedder(torch.tensor([0]))

    with pytest.raises(IndexError):  # or ValueError, depending on how you implement error checking
        action_embedder(torch.tensor([8]))


def test_deterministic_output(game: Connect4Game, serializer: Connect4Serializer) -> None:
    state = game.initial_state()
    encoded_state = serializer.state_to_tensor(game, state)
    encoded_state_batch = encoded_state.unsqueeze(0)

    action = torch.tensor([1])

    state_embedder = Connect4StateEmbedder()
    action_embedder = Connect4ActionEmbedder()

    state_embedder.eval()
    action_embedder.eval()

    # Run embeddings twice and check if they're the same
    state_emb1 = state_embedder(encoded_state_batch)
    state_emb2 = state_embedder(encoded_state_batch)
    assert torch.allclose(state_emb1, state_emb2)

    action_emb1 = action_embedder(action)
    action_emb2 = action_embedder(action)
    assert torch.allclose(action_emb1, action_emb2)
