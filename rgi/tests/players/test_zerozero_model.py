from typing_extensions import override

import torch
import pytest
from rgi.players.zerozero.zerozero_model import ZeroZeroModel, zerozero_loss
from rgi.players.zerozero.zerozero_model import StateEmbedder, ActionEmbedder

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive

TAction = int
TGameState = tuple[int, int]


class DummyStateEmbedder(StateEmbedder[TGameState]):
    def __init__(self, embedding_dim: int = 64):
        super().__init__(embedding_dim)
        self.vocabulary = torch.randn(10, embedding_dim)  # Random embeddings for 10 integers

    @override
    def forward(self, state: TGameState) -> torch.Tensor:
        return (self.vocabulary[state[0]] + self.vocabulary[state[1]]) / 2


class DummyActionEmbedder(ActionEmbedder[TAction]):
    def __init__(self, embedding_dim: int = 64):
        super().__init__(embedding_dim)
        self.vocabulary = torch.randn(10, embedding_dim)  # Random embeddings for 10 integers

    @override
    def forward(self, action: TAction) -> torch.Tensor:
        return self.vocabulary[action]

    @override
    def all_action_embeddings(self) -> torch.Tensor:
        return self.vocabulary


@pytest.fixture
def model() -> ZeroZeroModel[TGameState, TAction]:
    return ZeroZeroModel(
        state_embedder=DummyStateEmbedder(),
        action_embedder=DummyActionEmbedder(),
        possible_actions=[0, 1, 2, 3, 4, 5, 6],
        embedding_dim=64,
        hidden_dim=128,
        shared_dim=256,
    )


def test_zerozero_model(model: ZeroZeroModel[TGameState, TAction]):
    state = (1, 5)
    value, policy_logits = model(state)

    assert isinstance(value, torch.Tensor)
    assert isinstance(policy_logits, torch.Tensor)
    assert value.shape == (1,)
    assert policy_logits.shape == (1, 7)  # 7 possible actions


def test_zerozero_loss(model: ZeroZeroModel[TGameState, TAction]):
    states = torch.tensor([(1, 5), (2, 3)], dtype=torch.float32)
    actions = torch.tensor([3, 1], dtype=torch.long)
    rewards = torch.tensor([0.5, -0.5], dtype=torch.float32)
    policy_targets = torch.tensor([[0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)

    loss, loss_dict = zerozero_loss(model, states, actions, rewards, policy_targets)

    assert isinstance(loss, torch.Tensor)
    assert isinstance(loss_dict, dict)
    assert "total_loss" in loss_dict
    assert "value_loss" in loss_dict
    assert "policy_loss" in loss_dict
