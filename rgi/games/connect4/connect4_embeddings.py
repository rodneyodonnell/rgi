import torch
import torch.nn as nn


class Connect4StateEmbedder(nn.Module):
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64 * 6 * 7, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.embedding_dim)

    def _state_to_array(self, encoded_state_batch: torch.Tensor) -> torch.Tensor:
        return encoded_state_batch[:, :-1].reshape(-1, 1, 6, 7)

    def forward(self, encoded_state_batch: torch.Tensor) -> torch.Tensor:
        x = self._state_to_array(encoded_state_batch)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Connect4ActionEmbedder(nn.Module):
    def __init__(self, embedding_dim: int = 64, num_actions: int = 7) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        self.embedding = nn.Embedding(num_actions, embedding_dim)

    def forward(self, action: int) -> torch.Tensor:
        return self.embedding(action - 1)

    def all_action_embeddings(self) -> torch.Tensor:
        return self.embedding.weight
