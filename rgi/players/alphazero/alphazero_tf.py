from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray
from tensorflow import Tensor
from tensorflow.keras import Model, layers, losses, optimizers

from rgi.core.archive import RowFileArchiver
from rgi.core.base import TAction, TGame, TGameState
from rgi.core.trajectory import GameTrajectory
from rgi.games.count21.count21 import Count21Game, Count21State
from rgi.players.alphazero.alphazero import MCTSData, PolicyValueNetwork

TState = TypeVar("TState")
TAction = TypeVar("TAction")
TPlayerData = TypeVar("TPlayerData")
TensorCompatible = Union[Tensor, str, float, NDArray[Any], int, Sequence[Any]]


class PVNetwork_Count21_TF(Model):  # type: ignore[type-arg]
    def __init__(self, state_dim: int, num_actions: int, num_players: int) -> None:
        """
        Args:
            state_dim: Size of the flattened input state.
            num_actions: Number of possible actions (output logits count).
            num_players: Number of players (size of the value vector).
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.num_players = num_players
        self.fc1 = layers.Dense(128, activation="relu")
        self.fc2 = layers.Dense(128, activation="relu")
        self.policy_head = layers.Dense(num_actions)  # logits, no activation
        self.value_head = layers.Dense(num_players, activation="tanh")  # values in (-1, 1)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for model serialization."""
        config = super().get_config()
        config.update(
            {
                "state_dim": self.state_dim,
                "num_actions": self.num_actions,
                "num_players": self.num_players,
            }
        )
        return config

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], custom_objects: Optional[Dict[str, Any]] = None
    ) -> PVNetwork_Count21_TF:
        """Create model instance from configuration."""
        return cls(
            state_dim=config["state_dim"],
            num_actions=config["num_actions"],
            num_players=config["num_players"],
        )

    def call(
        self,
        inputs: Tensor,
        training: Optional[bool] = None,
        mask: Optional[TensorCompatible] = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through the network.

        Args:
            inputs: Batch of state vectors.
            training: Whether we are training or not.
            mask: Optional mask tensor.

        Returns:
            Tuple of (policy_logits, value) tensors.
            - policy_logits: Batch of logits for each action.
            - value: Batch of value vectors (one per player).
        """
        x = self.fc1(inputs)
        x = self.fc2(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


class TFPVNetworkWrapper(PolicyValueNetwork[TGame, TGameState, TAction]):
    def __init__(self, tf_model: PVNetwork_Count21_TF) -> None:
        self.tf_model = tf_model

    def predict(
        self, game: TGame, state: TGameState, actions: Sequence[TAction]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Convert the given game state to a flat numpy array and perform a forward pass
        through the TF model. Returns a tuple (policy_logits, value) as numpy arrays.
        """
        # Example conversion: Assuming `state` is an object with `score` and `current_player` attributes.
        state_array = np.array([state.score, state.current_player], dtype=np.float32)  # type: ignore
        # Expand dimensions to create a batch with a single element.
        input_tensor = tf.convert_to_tensor(state_array.reshape(1, -1))
        policy_logits, value = self.tf_model(input_tensor, training=False)
        return policy_logits.numpy()[0], value.numpy()[0]


# A simple dataset to convert your self-play trajectories into training examples.
# Each data point is a tuple: (state, target_policy, target_value)
class TrajectoryDataset_Count21:
    def __init__(self, trajectories: Sequence[GameTrajectory[Count21State, int, MCTSData[int]]]) -> None:
        self.data: list[tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]] = []

        for traj in trajectories:
            final_reward: NDArray[np.float32] = np.array(traj.final_reward, dtype=np.float32)

            # For each state-action pair in the trajectory
            for i in range(len(traj.actions)):
                state = traj.game_states[i]
                state_array = np.array([state.score, state.current_player], dtype=np.float32)

                # Get MCTS visit counts for this state
                mcts_data = traj.player_data[i]

                # Use legal_actions from MCTSData instead of hardcoding the number of actions
                num_actions = len(mcts_data.legal_actions)
                visit_counts = np.zeros(num_actions, dtype=np.float32)
                total_visits = sum(mcts_data.policy_counts.values())

                # Convert visit counts to policy distribution
                for action, count in mcts_data.policy_counts.items():
                    # Find the index of this action in legal_actions
                    try:
                        action_idx = mcts_data.legal_actions.index(action)
                        visit_counts[action_idx] = count / total_visits
                    except ValueError:
                        # This should never happen if legal_actions is correctly maintained
                        raise ValueError(f"Action {action} not found in legal_actions {mcts_data.legal_actions}")

                self.data.append((state_array, visit_counts, final_reward))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        return self.data[idx]


def train_model(
    trajectories: Sequence[GameTrajectory[Count21State, int, MCTSData[int]]], num_epochs: int = 10, batch_size: int = 32
) -> PVNetwork_Count21_TF:
    dataset_obj: TrajectoryDataset_Count21 = TrajectoryDataset_Count21(trajectories)

    # Generator yielding (state, target_policy, target_value)
    def gen() -> Iterator[tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]]:
        for state, target_policy, target_value in dataset_obj.data:
            yield state, target_policy, target_value

    state_dim: int = dataset_obj.data[0][0].shape[0]
    num_actions: int = dataset_obj.data[0][1].shape[0]
    num_players: int = dataset_obj.data[0][2].shape[0]

    # Output signature for full probability distributions
    output_signature = (
        tf.TensorSpec(shape=(state_dim,), dtype=tf.float32),
        tf.TensorSpec(shape=(num_actions,), dtype=tf.float32),
        tf.TensorSpec(shape=(num_players,), dtype=tf.float32),
    )
    tf_dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    tf_dataset = tf_dataset.shuffle(buffer_size=1000).batch(batch_size)

    model = PVNetwork_Count21_TF(state_dim=state_dim, num_actions=num_actions, num_players=num_players)

    optimizer = optimizers.Adam(1e-3)
    # Use categorical crossentropy for policy loss since we have probability distributions
    policy_loss_fn = losses.CategoricalCrossentropy(from_logits=True)
    value_loss_fn = losses.MeanSquaredError()

    for epoch in range(num_epochs):
        epoch_policy_loss: float = 0.0
        epoch_value_loss: float = 0.0
        steps: int = 0
        for states, target_policies, target_values in tf_dataset:
            with tf.GradientTape() as tape:
                policy_logits, value_pred = model(states, training=True)
                policy_loss = policy_loss_fn(target_policies, policy_logits)
                value_loss = value_loss_fn(target_values, value_pred)
                total_loss = policy_loss + value_loss
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_policy_loss += float(policy_loss.numpy())
            epoch_value_loss += float(value_loss.numpy())
            steps += 1
        print(
            f"Epoch {epoch + 1}: "
            f"Policy Loss = {epoch_policy_loss / steps:.4f}, "
            f"Value Loss = {epoch_value_loss / steps:.4f}"
        )

    return model


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train Model.")
    parser.add_argument("--input", type=str, default="trajectories.npz", help="File to save the trajectories.")
    parser.add_argument("--output", type=str, default="tf_pv_network.weights.h5", help="File to save the model.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train.")
    args = parser.parse_args()

    # Replace with actual loading of your trajectories.
    archiver = RowFileArchiver()
    trajectories: Sequence[GameTrajectory[Count21State, int, MCTSData[int]]] = archiver.read_items(
        args.input, GameTrajectory
    )

    model = train_model(trajectories, num_epochs=args.num_epochs)
    model.save_weights(args.output)
    print(f"Model training complete. Weights saved to {args.output}")


if __name__ == "__main__":
    main()
