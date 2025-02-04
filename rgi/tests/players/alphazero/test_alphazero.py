import pytest
from rgi.players.alphazero.alphazero import MCTS, DummyPolicyValueNetwork
from rgi.games.count21.count21 import Count21Game

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
