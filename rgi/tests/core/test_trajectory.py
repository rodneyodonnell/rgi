# rgi/tests/core/test_trajectory.py

import pytest
import jax.numpy as jnp

from rgi.core.trajectory import Trajectory, EncodedTrajectory, encode_trajectory, save_trajectories, load_trajectories
from rgi.games.connect4 import Connect4Game, Connect4State, Connect4Serializer


@pytest.fixture
def game():
    return Connect4Game()


@pytest.fixture
def serializer():
    return Connect4Serializer()


@pytest.fixture
def sample_trajectory(game):
    initial_state = game.initial_state()
    states = [initial_state]
    actions = [4, 3, 2]  # Sample actions
    state_rewards = [0.0, 0.0, 0.0]  # No intermediate rewards for Connect4
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


def test_trajectory_creation(sample_trajectory):
    assert len(sample_trajectory.states) == 4
    assert len(sample_trajectory.actions) == 3
    assert len(sample_trajectory.state_rewards) == 3
    assert len(sample_trajectory.player_ids) == 3
    assert len(sample_trajectory.final_rewards) == 2


def test_encode_trajectory(game, serializer, sample_trajectory):
    encoded = encode_trajectory(game, sample_trajectory, serializer)

    assert isinstance(encoded, EncodedTrajectory)
    assert encoded.states.shape == (4, 43)  # 4 states, 43 elements per state (6*7 + 1)
    assert encoded.actions.shape == (3,)
    assert encoded.state_rewards.shape == (3,)
    assert encoded.player_ids.shape == (3,)
    assert encoded.final_rewards.shape == (2,)
    assert encoded.length == 4


def test_save_and_load_trajectories(tmp_path, game, serializer, sample_trajectory):
    encoded = encode_trajectory(game, sample_trajectory, serializer)
    file_path = tmp_path / "test_trajectories.npy"

    # Save trajectories
    save_trajectories([encoded], str(file_path))

    # Load trajectories
    loaded_trajectories = load_trajectories(str(file_path))

    assert len(loaded_trajectories) == 1
    loaded = loaded_trajectories[0]

    assert jnp.array_equal(loaded.states, encoded.states)
    assert jnp.array_equal(loaded.actions, encoded.actions)
    assert jnp.array_equal(loaded.state_rewards, encoded.state_rewards)
    assert jnp.array_equal(loaded.player_ids, encoded.player_ids)
    assert jnp.array_equal(loaded.final_rewards, encoded.final_rewards)
    assert loaded.length == encoded.length


def test_multiple_trajectories(tmp_path, game, serializer, sample_trajectory):
    encoded1 = encode_trajectory(game, sample_trajectory, serializer)
    encoded2 = encode_trajectory(game, sample_trajectory, serializer)  # Using the same trajectory for simplicity
    file_path = tmp_path / "test_multiple_trajectories.npy"

    # Save trajectories
    save_trajectories([encoded1, encoded2], str(file_path))

    # Load trajectories
    loaded_trajectories = load_trajectories(str(file_path))

    assert len(loaded_trajectories) == 2
    for loaded, original in zip(loaded_trajectories, [encoded1, encoded2]):
        assert jnp.array_equal(loaded.states, original.states)
        assert jnp.array_equal(loaded.actions, original.actions)
        assert jnp.array_equal(loaded.state_rewards, original.state_rewards)
        assert jnp.array_equal(loaded.player_ids, original.player_ids)
        assert jnp.array_equal(loaded.final_rewards, original.final_rewards)
        assert loaded.length == original.length


def test_variable_length_trajectories(tmp_path, game, serializer):
    trajectory1 = Trajectory(
        states=[game.initial_state() for _ in range(3)],
        actions=[1, 2],
        state_rewards=[0.0, 0.0],
        player_ids=[1, 2],
        final_rewards=[1.0, -1.0],
    )
    trajectory2 = Trajectory(
        states=[game.initial_state() for _ in range(5)],
        actions=[1, 2, 3, 4],
        state_rewards=[0.0, 0.0, 0.0, 0.0],
        player_ids=[1, 2, 1, 2],
        final_rewards=[1.0, -1.0],
    )

    encoded1 = encode_trajectory(game, trajectory1, serializer)
    encoded2 = encode_trajectory(game, trajectory2, serializer)

    file_path = tmp_path / "test_variable_length_trajectories.npy"

    # Save trajectories
    save_trajectories([encoded1, encoded2], str(file_path))

    # Load trajectories
    loaded_trajectories = load_trajectories(str(file_path))

    assert len(loaded_trajectories) == 2
    assert loaded_trajectories[0].length == 3
    assert loaded_trajectories[1].length == 5

    for loaded, original in zip(loaded_trajectories, [encoded1, encoded2]):
        assert jnp.array_equal(loaded.states, original.states)
        assert jnp.array_equal(loaded.actions, original.actions)
        assert jnp.array_equal(loaded.state_rewards, original.state_rewards)
        assert jnp.array_equal(loaded.player_ids, original.player_ids)
        assert jnp.array_equal(loaded.final_rewards, original.final_rewards)
        assert loaded.length == original.length
