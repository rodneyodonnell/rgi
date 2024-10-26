import torch
from torch import nn

from rgi.games import connect4


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

    def forward(self, game_states: connect4.BatchGameState) -> torch.Tensor:
        # Add an extra dimension for the channel. Use (N, C, H, W) format.
        x = game_states.board.unsqueeze(1).float()
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

    def forward(self, actions: connect4.BatchAction) -> torch.Tensor:
        return self.embedding(actions.values - 1)

    def all_action_embeddings(self) -> torch.Tensor:
        return self.embedding.weight
