from typing import Any
from typing_extensions import override
import jax
import jax.numpy as jnp
import pytest
from rgi.players.zerozero.zerozero_model import ZeroZeroModel, zerozero_loss
from rgi.core.base import StateEmbedder, ActionEmbedder

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive

TAction = int
TEmbedding = jax.Array


class DummyStateEmbedder(StateEmbedder[Any, TEmbedding]):
    @override
    def embed_state(self, state: Any) -> TEmbedding:
        return jnp.array(state)

    @override
    def get_embedding_dim(self) -> int:
        return 64


class DummyActionEmbedder(ActionEmbedder[TAction, TEmbedding]):
    @override
    def embed_action(self, action: TAction) -> TEmbedding:
        return jnp.zeros(64).at[action].set(1)

    @override
    def get_embedding_dim(self) -> int:
        return 64


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
    dummy_state = jnp.zeros(64)
    dummy_state = dummy_state.at[0].set(999)
    dummy_action = 2
    init_params = model.init(key, dummy_state, dummy_action)
    return dict(init_params)


def test_model_output_shapes(model: ZeroZeroModel[Any, Any, TAction], params: dict[str, Any]) -> None:
    state: jax.Array = jnp.zeros(64)
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
    state: jax.Array = jnp.zeros(64)
    action: TAction = 5
    next_state: jax.Array = jnp.ones(64)
    reward: float = 1.0
    policy_target: jax.Array = jnp.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1])

    total_loss, loss_dict = zerozero_loss(params, model, state, action, next_state, reward, policy_target)

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
