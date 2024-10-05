from typing import Any
from typing_extensions import override
import jax
import jax.numpy as jnp
import pytest
from rgi.players.zerozero.zerozero_model import ZeroZeroModel, zerozero_loss
from rgi.players.zerozero.zerozero_model import StateEmbedder, ActionEmbedder
from flax import linen as nn

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive

TAction = int
TGameState = tuple[int, int]


# Dummy embedders for testing
class DummyStateEmbedder(StateEmbedder[TGameState]):
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, state: TGameState) -> jax.Array:
        return jnp.zeros(64).at[state[0]].set(10).at[state[1]].set(20)


class DummyActionEmbedder(ActionEmbedder[TAction]):
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, action: TAction) -> jax.Array:
        return jnp.zeros(64).at[action].set(1)


@pytest.fixture
def model() -> ZeroZeroModel[Any, Any, TAction]:
    return ZeroZeroModel(
        state_embedder=DummyStateEmbedder(),
        action_embedder=DummyActionEmbedder(),
        possible_actions=[0, 1, 2, 3, 4, 5, 6],
        embedding_dim=64,
        hidden_dim=128,
    )


@pytest.fixture
def params(model: ZeroZeroModel[Any, Any, TAction]) -> dict[str, Any]:
    key = jax.random.PRNGKey(0)
    dummy_state: TGameState = (1, 5)
    dummy_action: TAction = 2
    init_params = model.init(key, dummy_state, dummy_action)
    return dict(init_params)


def test_model_output_shapes(model: ZeroZeroModel[Any, Any, TAction], params: dict[str, Any]) -> None:
    state: TGameState = (1, 2)
    action: TAction = 5

    output = model.apply(params, state, action)
    next_state, reward, policy_embedding = output  # type: ignore

    assert next_state.shape == (64,)
    assert isinstance(reward, float)
    assert policy_embedding.shape == (64,)


def test_compute_action_probabilities(model: ZeroZeroModel[Any, Any, TAction], params: dict[str, Any]) -> None:
    policy_embedding: jax.Array = jnp.ones(64)

    action_probs = model.apply(
        params,
        method=model.compute_action_probabilities,
        policy_embedding=policy_embedding,
    )
    assert isinstance(action_probs, jax.Array)

    assert action_probs.shape == (7,)
    assert jnp.isclose(jnp.sum(action_probs), 1.0)


def test_zerozero_loss(model: ZeroZeroModel[Any, Any, TAction], params: dict[str, Any]) -> None:
    state: TGameState = (1, 2)
    action: TAction = 5
    next_state: TGameState = (1, 3)
    reward: float = 1.0
    policy_target: jax.Array = jnp.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1])

    total_loss, loss_dict = zerozero_loss(model, params, state, action, next_state, reward, policy_target)

    assert isinstance(total_loss, jax.Array)
    assert total_loss.shape == ()
    assert set(loss_dict.keys()) == {
        "total_loss",
        "dynamics_loss",
        "reward_loss",
        "policy_loss",
    }
    assert all(isinstance(v, jax.Array) and (v.shape == ()) for v in loss_dict.values())


if __name__ == "__main__":
    pytest.main([__file__])
