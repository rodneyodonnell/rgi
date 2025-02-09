from typing import cast, Any, Type
from pathlib import Path

import numpy as np
import tensorflow as tf
import pytest

from rgi.games.count21.count21 import Count21Game, Count21State
from rgi.players.alphazero.alphazero import AlphaZeroPlayer, DummyPolicyValueNetwork
from rgi.players.alphazero.alphazero_tf import PVNetwork_Count21_TF, TFPVNetworkWrapper

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive

# Assuming Count21Game returns a fixed shape state, e.g. np.array([score, current_player])
# and has a fixed action space size (e.g., 3 actions).


@pytest.fixture
def count21_game() -> Count21Game:
    return Count21Game(num_players=2, target=21, max_guess=3)


@pytest.fixture
def loaded_tf_model(count21_game: Count21Game) -> PVNetwork_Count21_TF:
    # Assume the game state can be converted to a flat state vector.
    state = count21_game.initial_state()
    # For our example, assume state has two features: score and current_player.
    state_np = np.array([state.score, state.current_player], dtype=np.float32)
    state_dim = state_np.shape[0]
    num_actions = len(count21_game.legal_actions(state))
    num_players = count21_game.num_players(state)

    model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)
    # Build the model using a dummy forward pass.
    model(tf.convert_to_tensor(np.expand_dims(state_np, axis=0)))
    model.load_weights("tf_pv_network.weights.h5")
    return model


@pytest.fixture
def improved_player(count21_game: Count21Game, loaded_tf_model: PVNetwork_Count21_TF) -> AlphaZeroPlayer:
    # Wrap the TF model so it implements the expected PolicyValueNetwork interface.
    wrapped_model = TFPVNetworkWrapper(loaded_tf_model)
    return AlphaZeroPlayer(count21_game, wrapped_model)


def test_model_prediction_shapes(count21_game: Count21Game, loaded_tf_model: PVNetwork_Count21_TF) -> None:
    state = count21_game.initial_state()
    state_array = np.array([state.score, state.current_player], dtype=np.float32)
    input_tensor = tf.convert_to_tensor(np.expand_dims(state_array, axis=0))
    policy_logits, value = loaded_tf_model(input_tensor, training=False)
    assert policy_logits.shape[0] == 1
    assert value.shape[0] == 1
    num_actions = len(count21_game.legal_actions(state))
    num_players = count21_game.num_players(state)
    assert policy_logits.shape[1] == num_actions
    assert value.shape[1] == num_players


def test_player_select_action(improved_player: AlphaZeroPlayer, count21_game: Count21Game) -> None:
    state = count21_game.initial_state()
    legal_actions = count21_game.legal_actions(state)
    action = improved_player.select_action(state, legal_actions)
    assert action in legal_actions


def test_trained_model() -> None:
    count21_game = Count21Game(num_players=2, target=8, max_guess=3)

    # Assume the game state can be converted to a flat state vector.
    state = count21_game.initial_state()
    # For our example, assume state has two features: score and current_player.
    state_np = np.array([state.score, state.current_player], dtype=np.float32)
    state_dim = state_np.shape[0]
    num_actions = len(count21_game.legal_actions(state))
    num_players = count21_game.num_players(state)

    model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)
    model(tf.convert_to_tensor(np.expand_dims(state_np, axis=0)))
    model.load_weights("tf_pv_network.g1000.s150.t8.weights.h5")

    wrapped_model = TFPVNetworkWrapper(model)
    dummy_model = DummyPolicyValueNetwork()

    player_100 = AlphaZeroPlayer(count21_game, wrapped_model, num_simulations=100)
    action_100 = player_100.select_action(state, count21_game.legal_actions(state))
    assert action_100 in count21_game.legal_actions(state)

    player_100_d = AlphaZeroPlayer(count21_game, dummy_model, num_simulations=100)
    action_100_d = player_100_d.select_action(state, count21_game.legal_actions(state))
    assert action_100_d in count21_game.legal_actions(state)

    player_1000 = AlphaZeroPlayer(count21_game, wrapped_model, num_simulations=1000)
    action_1000 = player_1000.select_action(state, count21_game.legal_actions(state))
    assert action_1000 in count21_game.legal_actions(state)

    player_1000_d = AlphaZeroPlayer(count21_game, dummy_model, num_simulations=1000)
    action_1000_d = player_1000_d.select_action(state, count21_game.legal_actions(state))
    assert action_1000_d in count21_game.legal_actions(state)

    player_5000 = AlphaZeroPlayer(count21_game, wrapped_model, num_simulations=5000)
    action_5000 = player_5000.select_action(state, count21_game.legal_actions(state))
    assert action_5000 in count21_game.legal_actions(state)

    player_5000_d = AlphaZeroPlayer(count21_game, dummy_model, num_simulations=5000)
    action_5000_d = player_5000_d.select_action(state, count21_game.legal_actions(state))
    assert action_5000_d in count21_game.legal_actions(state)

    player_10000 = AlphaZeroPlayer(count21_game, wrapped_model, num_simulations=10000)
    action_10000 = player_10000.select_action(state, count21_game.legal_actions(state))
    assert action_10000 in count21_game.legal_actions(state)

    player_10000_d = AlphaZeroPlayer(count21_game, dummy_model, num_simulations=10000)
    action_10000_d = player_10000_d.select_action(state, count21_game.legal_actions(state))
    assert action_10000_d in count21_game.legal_actions(state)


def test_tf_model_basic() -> None:
    # Create a simple model for testing.
    state_dim = 2  # score and current_player
    num_actions = 3  # 1, 2, 3
    num_players = 2

    model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)

    # Test forward pass.
    batch_size = 4
    inputs = tf.random.normal((batch_size, state_dim))
    policy_logits, value = model(inputs, training=False)

    assert policy_logits.shape == (batch_size, num_actions)
    assert value.shape == (batch_size, num_players)


def test_tf_model_with_alphazero() -> None:
    # Create a simple model for testing.
    state_dim = 2  # score and current_player
    num_actions = 3  # 1, 2, 3
    num_players = 2

    model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)
    wrapper = TFPVNetworkWrapper(model)

    game = Count21Game(num_players=num_players, target=21, max_guess=3)
    player = AlphaZeroPlayer[Count21Game, Count21State, int](game, wrapper, num_simulations=10)

    # Test that we can use the model to make predictions.
    state = Count21State(score=0, current_player=1)
    legal_actions = game.legal_actions(state)
    action = player.select_action(state, legal_actions)

    assert isinstance(action, int)
    assert action in legal_actions


def test_tf_model_save_load(tmp_path: Path) -> None:
    # Create a simple model for testing.
    state_dim = 2  # score and current_player
    num_actions = 3  # 1, 2, 3
    num_players = 2

    model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)
    wrapper = TFPVNetworkWrapper(model)

    game = Count21Game(num_players=num_players, target=21, max_guess=3)
    player = AlphaZeroPlayer[Count21Game, Count21State, int](game, wrapper, num_simulations=10)

    # Generate some random inputs.
    state = Count21State(score=0, current_player=1)
    legal_actions = game.legal_actions(state)

    # Get predictions before saving.
    network = cast(TFPVNetworkWrapper, player.network)
    dummy_model = network.tf_model
    input_tensor = tf.convert_to_tensor(np.array([[0, 1]], dtype=np.float32))
    before_policy, before_value = dummy_model(input_tensor, training=False)

    # Save the model.
    weights_path = tmp_path / "test_model.weights.h5"
    dummy_model.save_weights(str(weights_path))

    # Create a new model and load the weights.
    new_model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)
    # Build the model by calling it with a dummy input
    new_model(input_tensor, training=False)
    new_model.load_weights(str(weights_path))

    # Get predictions after loading.
    after_policy, after_value = new_model(input_tensor, training=False)

    # Check that the predictions are the same.
    np.testing.assert_array_almost_equal(before_policy.numpy(), after_policy.numpy())
    np.testing.assert_array_almost_equal(before_value.numpy(), after_value.numpy())
