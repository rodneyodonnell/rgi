"""Tests for AlphaZero TensorFlow model training."""

import numpy as np
import tensorflow as tf

from rgi.core.game_runner import GameRunner
from rgi.core.trajectory import GameTrajectory
from rgi.games.count21.count21 import Count21Game, Count21State
from rgi.players.alphazero.alphazero import AlphaZeroPlayer, MCTSData
from rgi.players.alphazero.alphazero_tf import PVNetwork_Count21_TF, TFPVNetworkWrapper, train_model
from rgi.players.random_player.random_player import RandomPlayer


def test_training_pipeline() -> None:
    """Test the full training pipeline from self-play to trained model."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Create game and initial model
    game = Count21Game(num_players=2, target=21, max_guess=3)
    state = game.initial_state()
    state_array = np.array([state.score, state.current_player], dtype=np.float32)
    state_dim = state_array.shape[0]
    num_actions = len(game.legal_actions(state))
    num_players = game.num_players(state)

    initial_model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)
    # Build model via a dummy forward pass
    initial_model(tf.convert_to_tensor(state_array.reshape(1, -1)))
    wrapped_model = TFPVNetworkWrapper(initial_model)

    # Generate self-play games
    num_games = 10
    trajectories: list[GameTrajectory[Count21State, int, MCTSData[int]]] = []

    for _ in range(num_games):
        players = [AlphaZeroPlayer(game, wrapped_model, num_simulations=50) for _ in range(2)]
        runner = GameRunner(game, players, verbose=False)
        trajectory = runner.run()
        trajectories.append(trajectory)

    # Train model
    trained_model = train_model(trajectories, num_epochs=20, batch_size=32)
    trained_wrapper = TFPVNetworkWrapper(trained_model)

    # Evaluate against random player
    num_eval_games = 100
    alphazero_wins = 0

    for _ in range(num_eval_games):
        players = [
            AlphaZeroPlayer(game, trained_wrapper, num_simulations=50),
            RandomPlayer(),
        ]
        runner = GameRunner(game, players, verbose=False)
        trajectory = runner.run()
        if trajectory.final_reward[0] > 0:  # AlphaZero player won
            alphazero_wins += 1

    win_rate = alphazero_wins / num_eval_games
    print(f"\nAlphaZero win rate against random player: {win_rate:.2%}")

    # The policy should be strong enough to consistently beat random
    assert win_rate > 0.8, f"Win rate {win_rate:.2%} is too low against random player"

    # Check policy outputs for key states
    test_states = [
        (0, 1),  # Start of game
        (18, 1),  # Near target, player 1's turn
        (19, 2),  # Near target, player 2's turn
    ]

    print("\nLearned policies for key states:")
    for score, player in test_states:
        state = Count21State(score=score, current_player=player)
        policy_logits, value = trained_wrapper.predict(game, state, game.legal_actions(state))
        policy = np.exp(policy_logits) / np.sum(np.exp(policy_logits))
        print(f"\nState: score={score}, player={player}")
        print(f"Policy: {policy}")
        print(f"Value: {value}")

        # Basic sanity checks on the policy
        assert np.allclose(np.sum(policy), 1.0), "Policy should be a valid probability distribution"
        assert np.all(policy >= 0), "Policy probabilities should be non-negative"
        assert np.all(value >= -1) and np.all(value <= 1), "Values should be in [-1, 1]"
