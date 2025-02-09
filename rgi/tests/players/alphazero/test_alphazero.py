import pytest
from rgi.players.alphazero.alphazero import MCTS, DummyPolicyValueNetwork, AlphaZeroPlayer
from rgi.games.count21.count21 import Count21Game, Count21State
from typing import cast, Any, Type
import numpy as np

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


# For faster tests, we set a low target (e.g. 5) so that terminal states are reached quickly.
@pytest.fixture
def count21_two_player_game() -> Count21Game:
    # Two-player Count21 game with target 5 and max_guess 3.
    return Count21Game(num_players=2, target=5, max_guess=3)


@pytest.fixture
def count21_three_player_game() -> Count21Game:
    # Three-player Count21 game with target 5 and max_guess 3.
    return Count21Game(num_players=3, target=5, max_guess=3)


@pytest.fixture
def dummy_network() -> DummyPolicyValueNetwork:
    # Using the dummy network available from alphazero.
    return DummyPolicyValueNetwork()


def test_mcts_search_count21_two_player(
    count21_two_player_game: Count21Game, dummy_network: DummyPolicyValueNetwork
) -> None:
    mcts = MCTS(count21_two_player_game, dummy_network, c_puct=1.0, num_simulations=10)
    state = count21_two_player_game.initial_state()
    action_visits = mcts.search(state)
    # Verify that each legal action has been expanded with non-negative visit counts.
    legal_actions = count21_two_player_game.legal_actions(state)
    for action in legal_actions:
        assert action in action_visits, f"Action {action} missing in visit counts."
        assert action_visits[action] >= 0, f"Action {action} has negative visits."


def test_mcts_search_count21_three_player(
    count21_three_player_game: Count21Game, dummy_network: DummyPolicyValueNetwork
) -> None:
    mcts = MCTS(count21_three_player_game, dummy_network, c_puct=1.0, num_simulations=10)
    state = count21_three_player_game.initial_state()
    action_visits = mcts.search(state)
    # Verify that each legal action was expanded.
    legal_actions = count21_three_player_game.legal_actions(state)
    for action in legal_actions:
        assert action in action_visits, f"Action {action} missing in visit counts."
        assert action_visits[action] >= 0, f"Action {action} has negative visits."


def test_mcts_search_count21_two_player_optimal_play(dummy_network: DummyPolicyValueNetwork) -> None:
    game = Count21Game(num_players=2, target=11, max_guess=3)

    mcts = MCTS(game, dummy_network, c_puct=1.0, num_simulations=2000)
    initial_state = game.initial_state()
    action_visits = mcts.search(initial_state)

    # Optimal play is action '2'
    assert action_visits[1] < 500
    assert action_visits[2] > 1000
    assert action_visits[3] < 500


def test_dummy_policy_value_network() -> None:
    game = Count21Game()
    state = Count21State(score=0, current_player=1)
    actions = game.legal_actions(state)
    network = DummyPolicyValueNetwork[Count21Game, Count21State, int]()

    policy_logits, value = network.predict(game, state, actions)

    assert isinstance(policy_logits, np.ndarray)
    assert isinstance(value, np.ndarray)
    assert policy_logits.shape == (len(actions),)
    assert value.shape == (game.num_players(state),)


def test_alphazero_player() -> None:
    game = Count21Game()
    network = DummyPolicyValueNetwork[Count21Game, Count21State, int]()
    player = AlphaZeroPlayer[Count21Game, Count21State, int](game, network, num_simulations=10)

    state = Count21State(score=0, current_player=1)
    actions = game.legal_actions(state)
    action = player.select_action(state, actions)

    assert isinstance(action, int)
    assert action in actions


def test_alphazero_player_with_different_simulations() -> None:
    game = Count21Game()
    network = DummyPolicyValueNetwork[Count21Game, Count21State, int]()

    player_10 = AlphaZeroPlayer[Count21Game, Count21State, int](game, network, num_simulations=10)
    player_100 = AlphaZeroPlayer[Count21Game, Count21State, int](game, network, num_simulations=100)

    state = Count21State(score=0, current_player=1)
    action_10 = player_10.select_action(state, game.legal_actions(state))
    action_100 = player_100.select_action(state, game.legal_actions(state))

    assert action_10 in game.legal_actions(state)
    assert action_100 in game.legal_actions(state)
