import pytest
import jax
import jax.numpy as jnp
from flax.training import train_state
from rgi.players.zerozero.zerozero_model import ZeroZeroModel
from rgi.players.zerozero.zerozero_trainer import ZeroZeroTrainer
from rgi.core.trajectory import EncodedTrajectory
from rgi.tests.players.test_zerozero_model import DummyStateEmbedder, DummyActionEmbedder
from rgi.core.trajectory import save_trajectories


@pytest.fixture
def dummy_model():
    return ZeroZeroModel(
        state_embedder=DummyStateEmbedder(),
        action_embedder=DummyActionEmbedder(),
        possible_actions=[0, 1, 2],
        embedding_dim=64,
        hidden_dim=8,
        shared_dim=16,
    )


@pytest.fixture
def dummy_trainer(dummy_model):
    return ZeroZeroTrainer(dummy_model)


@pytest.fixture
def dummy_trajectories():
    return [
        EncodedTrajectory(
            states=jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
            actions=jnp.array([1, 2, 0]),
            state_rewards=jnp.array([0, 0, 1]),
            player_ids=jnp.array([1, 2, 1]),
            final_rewards=jnp.array([1, -1]),
            length=3,
        )
    ]


def test_create_train_state(dummy_trainer):
    rng = jax.random.PRNGKey(0)
    train_state = dummy_trainer.create_train_state(rng)
    assert train_state is not None
    assert train_state.params is not None
    assert train_state.opt_state is not None


def test_train_step(dummy_trainer):
    rng = jax.random.PRNGKey(0)
    train_state = dummy_trainer.create_train_state(rng)
    batch = (
        jnp.array([[1, 0, 0, 0]]),
        jnp.array([1]),
        jnp.array([[0, 1, 0, 0]]),
        jnp.array([0.0]),
        jnp.array([[0, 1, 0]]),
    )
    new_train_state, loss_dict = dummy_trainer.train_step(train_state, batch)
    assert new_train_state is not None
    assert loss_dict is not None
    assert "total_loss" in loss_dict


def test_create_batches(dummy_trainer, dummy_trajectories):
    batches = list(dummy_trainer.create_batches(dummy_trajectories, batch_size=2))
    assert len(batches) == 1
    batch = batches[0]
    assert len(batch) == 5
    assert batch[0].shape == (2, 4)  # states
    assert batch[1].shape == (2,)  # actions
    assert batch[2].shape == (2, 4)  # next_states
    assert batch[3].shape == (2,)  # rewards
    assert batch[4].shape == (2, 3)  # policy_targets


def test_train(dummy_trainer, dummy_trajectories, tmp_path):
    trajectories_file = tmp_path / "test_trajectories.npy"
    save_trajectories(dummy_trajectories, str(trajectories_file))

    dummy_trainer.train(str(trajectories_file), num_epochs=1, batch_size=2)
    assert dummy_trainer.state is not None


def test_save_load_checkpoint(dummy_trainer, dummy_trajectories, tmp_path):
    trajectories_file = tmp_path / "test_trajectories.npy"
    save_trajectories(dummy_trajectories, str(trajectories_file))

    dummy_trainer.train(str(trajectories_file), num_epochs=1, batch_size=2)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    dummy_trainer.save_checkpoint(str(checkpoint_dir))
    new_trainer = ZeroZeroTrainer(dummy_trainer.model)
    new_trainer.load_checkpoint(str(checkpoint_dir))

    assert new_trainer.state is not None
    assert isinstance(new_trainer.state, train_state.TrainState)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), dummy_trainer.state.params, new_trainer.state.params)
    )

    # Check that the optimizer states have the same structure
    assert jax.tree_util.tree_structure(dummy_trainer.state.opt_state) == jax.tree_util.tree_structure(
        new_trainer.state.opt_state
    )

    # Check that the optimizer states have similar values
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda x, y: jnp.allclose(x, y) if isinstance(x, jnp.ndarray) else x == y,
            dummy_trainer.state.opt_state,
            new_trainer.state.opt_state,
        )
    )
