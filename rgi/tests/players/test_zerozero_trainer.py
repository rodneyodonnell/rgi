import torch
import pytest
from typing import Any, List
from rgi.players.zerozero.zerozero_trainer import ZeroZeroTrainer
from rgi.players.zerozero.zerozero_model import ZeroZeroModel
from rgi.core.base import GameSerializer, Game
from rgi.core.trajectory import EncodedTrajectory
from rgi.tests.players.test_zerozero_model import DummyStateEmbedder, DummyActionEmbedder


class DummyGame(Game[tuple[int, int], int]):
    def all_actions(self) -> List[int]:
        return [0, 1, 2]


class DummySerializer(GameSerializer[tuple[int, int], int]):
    def state_to_tensor(self, game: Game[tuple[int, int], int], state: tuple[int, int]) -> torch.Tensor:
        return torch.tensor([state[0], state[1], 0, 0], dtype=torch.float32)


@pytest.fixture
def dummy_model() -> ZeroZeroModel[Any, Any, int]:
    return ZeroZeroModel(
        state_embedder=DummyStateEmbedder(),
        action_embedder=DummyActionEmbedder(),
        possible_actions=[0, 1, 2],
        embedding_dim=64,
        hidden_dim=8,
        shared_dim=16,
    )


@pytest.fixture
def dummy_trainer(dummy_model: ZeroZeroModel[Any, Any, int]) -> ZeroZeroTrainer:
    return ZeroZeroTrainer(dummy_model, DummySerializer(), DummyGame())


@pytest.fixture
def dummy_trajectories() -> List[EncodedTrajectory]:
    return [
        EncodedTrajectory(
            states=torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32),
            actions=torch.tensor([1, 2, 0], dtype=torch.long),
            state_rewards=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
            player_ids=torch.tensor([1, 2, 1], dtype=torch.long),
            final_rewards=torch.tensor([1.0, -1.0], dtype=torch.float32),
            length=3,
        )
    ]


def test_create_batches(dummy_trainer: ZeroZeroTrainer, dummy_trajectories: List[EncodedTrajectory]):
    dataloader = dummy_trainer.create_batches(dummy_trajectories, batch_size=2)
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    for batch in dataloader:
        assert len(batch) == 4
        states, actions, rewards, policy_targets = batch
        assert states.shape == (2, 4)
        assert actions.shape == (2,)
        assert rewards.shape == (2,)
        assert policy_targets.shape == (2, 3)


def test_train_step(dummy_trainer: ZeroZeroTrainer, dummy_trajectories: List[EncodedTrajectory]):
    dataloader = dummy_trainer.create_batches(dummy_trajectories, batch_size=2)
    for batch in dataloader:
        loss, loss_dict = dummy_trainer.train_step(batch)
        assert isinstance(loss, float)
        assert isinstance(loss_dict, dict)
        assert "total_loss" in loss_dict
        assert "value_loss" in loss_dict
        assert "policy_loss" in loss_dict
        break


def test_train(dummy_trainer: ZeroZeroTrainer, dummy_trajectories: List[EncodedTrajectory]):
    dummy_trainer.train(dummy_trajectories, num_epochs=1, batch_size=2)


def test_save_load_checkpoint(dummy_trainer: ZeroZeroTrainer, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    dummy_trainer.save_checkpoint(str(checkpoint_dir))
    assert (checkpoint_dir / "zerozero_model.pth").exists()
    assert (checkpoint_dir / "zerozero_optimizer.pth").exists()

    new_trainer = ZeroZeroTrainer(dummy_trainer.model, DummySerializer(), DummyGame())
    new_trainer.load_checkpoint(str(checkpoint_dir))

    assert torch.allclose(next(dummy_trainer.model.parameters()), next(new_trainer.model.parameters()))
