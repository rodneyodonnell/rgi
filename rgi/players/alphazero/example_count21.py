#!/usr/bin/env python3
"""
Example script demonstrating how to use AlphaZero with the Count21 game.
This script shows how to:
1. Create a model
2. Generate self-play data
3. Train the model
4. Evaluate the model
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from typing import List, Sequence

import numpy as np
import tensorflow as tf

from rgi.core.archive import RowFileArchiver
from rgi.core.trajectory import GameTrajectory
from rgi.games.count21.count21 import Count21Game, Count21State
from rgi.players.alphazero.alphazero import AlphaZeroPlayer, MCTSData
from rgi.players.alphazero.alphazero_tf import (
    PVNetwork_Count21_TF,
    TFPVNetworkWrapper,
    load_model,
    save_model,
    train_model,
)
from rgi.players.random import RandomPlayer


def create_initial_model() -> PVNetwork_Count21_TF:
    """Create an initial untrained model for Count21."""
    # For Count21, the state is represented as [score, current_player]
    state_dim = 2
    # Count21 has 3 possible actions (1, 2, 3)
    num_actions = 3
    # Count21 has 2 players
    num_players = 2

    model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)
    return model


def generate_self_play_data(
    model: PVNetwork_Count21_TF,
    num_games: int = 100,
    num_simulations: int = 50,
    temperature: float = 1.0,
) -> List[GameTrajectory[Count21State, int, MCTSData[int]]]:
    """Generate self-play data using the provided model."""
    game = Count21Game()
    network_wrapper = TFPVNetworkWrapper(model)

    # Create an AlphaZero player
    player = AlphaZeroPlayer(game, network_wrapper, num_simulations=num_simulations)

    trajectories: List[GameTrajectory[Count21State, int, MCTSData[int]]] = []

    for i in range(num_games):
        if i % 10 == 0:
            print(f"Playing game {i+1}/{num_games}")

        # Initialize a new trajectory
        trajectory = GameTrajectory[Count21State, int, MCTSData[int]](game_id=f"self_play_{i}")

        # Start a new game
        state = game.initial_state()
        trajectory.game_states.append(state)

        # Play until the game is over
        while not game.is_terminal(state):
            current_player_id = game.current_player_id(state)
            legal_actions = game.legal_actions(state)

            # Get action from AlphaZero player
            action_result = player.select_action(state, legal_actions)
            action = action_result.action

            # Store the action and player data
            trajectory.actions.append(action)
            trajectory.player_data.append(action_result.player_data)

            # Apply the action
            state = game.next_state(state, action)
            trajectory.game_states.append(state)

        # Store the final reward
        rewards = [game.reward(state, player_id) for player_id in range(1, game.num_players(state) + 1)]
        trajectory.final_reward = rewards

        # Add the trajectory to our collection
        trajectories.append(trajectory)

    return trajectories


def evaluate_model(
    model: PVNetwork_Count21_TF,
    opponent_model: PVNetwork_Count21_TF = None,
    num_games: int = 100,
    num_simulations: int = 50,
) -> float:
    """
    Evaluate a model by playing against a random player or another model.
    Returns the win rate of the model.
    """
    game = Count21Game()
    network_wrapper = TFPVNetworkWrapper(model)

    # Create an AlphaZero player with the model
    player1 = AlphaZeroPlayer(game, network_wrapper, num_simulations=num_simulations)

    # Create the opponent
    if opponent_model is not None:
        opponent_wrapper = TFPVNetworkWrapper(opponent_model)
        player2 = AlphaZeroPlayer(game, opponent_wrapper, num_simulations=num_simulations)
    else:
        player2 = RandomPlayer()

    # Track wins
    player1_wins = 0

    for i in range(num_games):
        if i % 10 == 0:
            print(f"Playing evaluation game {i+1}/{num_games}")

        # Randomly decide who goes first
        players = [player1, player2]
        random.shuffle(players)

        # Start a new game
        state = game.initial_state()

        # Play until the game is over
        while not game.is_terminal(state):
            current_player_id = game.current_player_id(state)
            current_player = players[current_player_id - 1]  # Convert 1-indexed to 0-indexed

            legal_actions = game.legal_actions(state)
            action_result = current_player.select_action(state, legal_actions)
            action = action_result.action

            state = game.next_state(state, action)

        # Determine winner
        if game.reward(state, 1 if players[0] == player1 else 2) > 0:
            if players[0] == player1:
                player1_wins += 1
        elif game.reward(state, 2 if players[0] == player1 else 1) > 0:
            if players[1] == player1:
                player1_wins += 1

    win_rate = player1_wins / num_games
    print(f"Evaluation complete. Model won {player1_wins}/{num_games} games ({win_rate:.2%})")
    return win_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero example for Count21")
    parser.add_argument("--output_dir", type=str, default="alphazero_count21", help="Directory to save outputs")
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of self-play/train iterations")
    parser.add_argument("--games_per_iteration", type=int, default=100, help="Number of self-play games per iteration")
    parser.add_argument("--num_simulations", type=int, default=50, help="Number of MCTS simulations per move")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs per iteration")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load an existing model")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create or load model
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model = load_model(args.load_model)
    else:
        print("Creating new model")
        model = create_initial_model()

    # Create archiver for saving trajectories
    archiver = RowFileArchiver()

    # Run iterations of self-play, training, and evaluation
    for iteration in range(args.num_iterations):
        print(f"\n=== Iteration {iteration+1}/{args.num_iterations} ===")

        # Create directory for this iteration
        iteration_dir = os.path.join(args.output_dir, f"iteration_{iteration+1}")
        os.makedirs(iteration_dir, exist_ok=True)

        # 1. Generate self-play data
        print(f"Generating {args.games_per_iteration} self-play games...")
        trajectories = generate_self_play_data(
            model, num_games=args.games_per_iteration, num_simulations=args.num_simulations
        )

        # Save trajectories
        trajectories_path = os.path.join(iteration_dir, "trajectories.npz")
        archiver.write_items(trajectories_path, trajectories)
        print(f"Saved {len(trajectories)} trajectories to {trajectories_path}")

        # 2. Train model
        print("Training model...")
        checkpoint_dir = os.path.join(iteration_dir, "checkpoints")
        model = train_model(
            trajectories,
            num_epochs=args.num_epochs,
            batch_size=32,
            validation_split=0.1,
            patience=3,
            checkpoint_dir=checkpoint_dir,
        )

        # Save model
        model_path = os.path.join(iteration_dir, "model")
        save_model(model, model_path)
        print(f"Saved model to {model_path}")

        # 3. Evaluate model
        print("Evaluating model against random player...")
        win_rate = evaluate_model(model, num_games=50, num_simulations=args.num_simulations)

        # Save evaluation results
        with open(os.path.join(iteration_dir, "evaluation.txt"), "w") as f:
            f.write(f"Win rate against random player: {win_rate:.2%}\n")

    print("\nAlphaZero training complete!")


if __name__ == "__main__":
    main()
