# rgi/tests/core/test_trajectory.py

from pathlib import Path
import pytest
import torch

from rgi.core.trajectory import Trajectory, EncodedTrajectory, encode_trajectory, save_trajectories, load_trajectories
from rgi.games.connect4 import Connect4Game, Connect4State, Connect4Serializer

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def game() -> Connect4Game:
    return Connect4Game()


@pytest.fixture
def serializer() -> Connect4Serializer:
    return Connect4Serializer()


@pytest.fixture
def sample_trajectory(game: Connect4Game) -> Trajectory:
    initial_state = game.initial_state()
    states = [initial_state]
    actions = [4, 3, 2]  # Sample actions
    state_rewards = [0.0, 0.0, 0.0, 0.0]  # No intermediate rewards for Connect4
    player_ids = [1, 2, 1]
    for action in actions:
        states.append(game.next_state(states[-1], action))

    return Trajectory(
        states=states,
        actions=actions,
        state_rewards=state_rewards,
        player_ids=player_ids,
        final_rewards=[1.0, -1.0],  # Player 1 wins, Player 2 loses
    )


def test_trajectory_creation(sample_trajectory: Trajectory) -> None:
    assert len(sample_trajectory.states) == 4
    assert len(sample_trajectory.actions) == 3
    assert len(sample_trajectory.state_rewards) == 4
    assert len(sample_trajectory.player_ids) == 3
    assert len(sample_trajectory.final_rewards) == 2


def test_encode_trajectory(game: Connect4Game, serializer: Connect4Serializer, sample_trajectory: Trajectory) -> None:
    encoded = encode_trajectory(game, sample_trajectory, serializer)

    assert isinstance(encoded, EncodedTrajectory)
    assert encoded.states.shape == (4, 43)  # 4 states, 43 elements per state (6*7 + 1)
    assert encoded.actions.shape == (3,)
    assert encoded.state_rewards.shape == (4,)
    assert encoded.player_ids.shape == (3,)
    assert encoded.final_rewards.shape == (2,)
    assert encoded.num_actions == 3
    assert encoded.num_players == 2


def test_save_and_load_trajectories(
    tmp_path: Path, game: Connect4Game, serializer: Connect4Serializer, sample_trajectory: Trajectory
) -> None:
    encoded = encode_trajectory(game, sample_trajectory, serializer)
    file_path = tmp_path / "test_trajectories.pt"

    # Save trajectories
    save_trajectories([encoded], str(file_path))

    # Load trajectories
    loaded_trajectories = load_trajectories(str(file_path))

    assert len(loaded_trajectories) == 1
    loaded = loaded_trajectories[0]

    assert torch.equal(loaded.states, encoded.states)
    assert torch.equal(loaded.actions, encoded.actions)
    assert torch.equal(loaded.state_rewards, encoded.state_rewards)
    assert torch.equal(loaded.player_ids, encoded.player_ids)
    assert torch.equal(loaded.final_rewards, encoded.final_rewards)
    assert loaded.num_actions == encoded.num_actions
    assert loaded.num_players == encoded.num_players


def test_multiple_trajectories(
    tmp_path: Path, game: Connect4Game, serializer: Connect4Serializer, sample_trajectory: Trajectory
) -> None:
    encoded1 = encode_trajectory(game, sample_trajectory, serializer)
    encoded2 = encode_trajectory(game, sample_trajectory, serializer)  # Using the same trajectory for simplicity
    all_encoded = [encoded1, encoded2]
    file_path = tmp_path / "test_multiple_trajectories.pt"

    # Save trajectories
    save_trajectories(all_encoded, str(file_path))

    # Load trajectories
    loaded_trajectories = load_trajectories(str(file_path))

    assert len(loaded_trajectories) == 2
    for loaded, original in zip(loaded_trajectories, all_encoded):
        assert torch.equal(loaded.states, original.states)
        assert torch.equal(loaded.actions, original.actions)
        assert torch.equal(loaded.state_rewards, original.state_rewards)
        assert torch.equal(loaded.player_ids, original.player_ids)
        assert torch.equal(loaded.final_rewards, original.final_rewards)
        assert loaded.num_actions == original.num_actions
        assert loaded.num_players == original.num_players


def test_variable_length_trajectories(tmp_path: Path, game: Connect4Game, serializer: Connect4Serializer) -> None:
    trajectory1 = Trajectory(
        states=[game.initial_state() for _ in range(3)],
        actions=[1, 2],
        state_rewards=[0.0, 0.0, 0.0],
        player_ids=[1, 2],
        final_rewards=[1.0, -1.0],
    )
    trajectory2 = Trajectory(
        states=[game.initial_state() for _ in range(5)],
        actions=[1, 2, 3, 4],
        state_rewards=[0.0, 0.0, 0.0, 0.0, 0.0],
        player_ids=[1, 2, 1, 2],
        final_rewards=[1.0, -1.0],
    )

    encoded1 = encode_trajectory(game, trajectory1, serializer)
    encoded2 = encode_trajectory(game, trajectory2, serializer)

    file_path = tmp_path / "test_variable_length_trajectories.pt"

    # Save trajectories
    save_trajectories([encoded1, encoded2], str(file_path))

    # Load trajectories
    loaded_trajectories = load_trajectories(str(file_path))

    assert len(loaded_trajectories) == 2
    assert loaded_trajectories[0].num_actions == 2
    assert loaded_trajectories[1].num_actions == 4

    for loaded, original in zip(loaded_trajectories, [encoded1, encoded2]):
        assert torch.equal(loaded.states, original.states)
        assert torch.equal(loaded.actions, original.actions)
        assert torch.equal(loaded.state_rewards, original.state_rewards)
        assert torch.equal(loaded.player_ids, original.player_ids)
        assert torch.equal(loaded.final_rewards, original.final_rewards)
        assert loaded.num_actions == original.num_actions
        assert loaded.num_players == original.num_players
