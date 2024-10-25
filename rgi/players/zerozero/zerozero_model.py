import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Generic, TypeVar, List
from rgi.core.base import TAction, TGameState

TState = TypeVar("TState", bound=TGameState)
TAction = TypeVar("TAction", bound=TAction)


class StateEmbedder(nn.Module, Generic[TState]):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, state: TState) -> torch.Tensor:
        raise NotImplementedError


class ActionEmbedder(nn.Module, Generic[TAction]):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, action: TAction) -> torch.Tensor:
        raise NotImplementedError

    def all_action_embeddings(self) -> torch.Tensor:
        raise NotImplementedError


class ZeroZeroModel(nn.Module, Generic[TState, TAction]):
    def __init__(
        self,
        state_embedder: StateEmbedder[TState],
        action_embedder: ActionEmbedder[TAction],
        possible_actions: List[TAction],
        embedding_dim: int,
        hidden_dim: int,
        shared_dim: int,
    ):
        super().__init__()
        self.state_embedder = state_embedder
        self.action_embedder = action_embedder
        self.possible_actions = possible_actions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.shared_dim = shared_dim

        self.shared_layer = nn.Linear(embedding_dim, shared_dim)
        self.value_head = nn.Linear(shared_dim, 1)
        self.policy_head = nn.Linear(shared_dim, len(possible_actions))

    def forward(self, state: TState) -> tuple[torch.Tensor, torch.Tensor]:
        state_embedding = self.state_embedder(state)
        shared_features = F.relu(self.shared_layer(state_embedding))
        value = self.value_head(shared_features)
        policy_logits = self.policy_head(shared_features)
        return value, policy_logits


def zerozero_loss(
    model: ZeroZeroModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    policy_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    values, policy_logits = model(states)

    value_loss = F.mse_loss(values.squeeze(), rewards)
    policy_loss = F.cross_entropy(policy_logits, actions)

    total_loss = value_loss + policy_loss

    return total_loss, {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "policy_loss": policy_loss,
    }
