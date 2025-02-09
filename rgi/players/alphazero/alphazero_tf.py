from typing import Sequence, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple
from rgi.games.count21.count21 import Count21Game, Count21State, Count21Action
from rgi.players.alphazero.alphazero import PolicyValueNetwork

from rgi.core.archive import RowFileArchiver
from rgi.core.trajectory import GameTrajectory


class PVNetwork_Count21_TF(keras.Model):
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

    def get_config(self) -> dict:
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
    def from_config(cls, config: dict) -> "PVNetwork_Count21_TF":
        """Create model instance from configuration."""
        return cls(
            state_dim=config["state_dim"],
            num_actions=config["num_actions"],
            num_players=config["num_players"],
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        x: tf.Tensor = self.fc1(inputs)
        x = self.fc2(x)
        policy_logits: tf.Tensor = self.policy_head(x)
        value: tf.Tensor = self.value_head(x)
        return policy_logits, value


class TFPVNetworkWrapper(PolicyValueNetwork[Count21Game, Count21State, Count21Action]):
    def __init__(self, tf_model: PVNetwork_Count21_TF) -> None:
        self.tf_model = tf_model

    def predict(self, game: Any, state: Any, actions: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the given game state to a flat numpy array and perform a forward pass
        through the TF model. Returns a tuple (policy_logits, value) as numpy arrays.
        """
        # Example conversion: Assuming `state` is an object with `score` and `current_player` attributes.
        state_array = np.array([state.score, state.current_player], dtype=np.float32)
        # Expand dimensions to create a batch with a single element.
        input_tensor = tf.convert_to_tensor(state_array.reshape(1, -1))
        policy_logits, value = self.tf_model(input_tensor, training=False)
        return policy_logits.numpy()[0], value.numpy()[0]


# A simple dataset to convert your self-play trajectories into training examples.
# Each data point is a tuple: (state, target_policy, target_value)
class TrajectoryDataset_Count21:
    def __init__(self, trajectories: Sequence[GameTrajectory[Count21State, Count21Action]]) -> None:
        self.data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for traj in trajectories:
            final_reward: np.ndarray = np.array(traj.final_reward, dtype=np.float32)
            # For each non-terminal state (skip last if terminal)
            for i in range(len(traj.game_states) - 1):
                state: np.ndarray = np.array(
                    [traj.game_states[i].score, traj.game_states[i].current_player], dtype=np.float32
                )
                # Here, instead of aggregating all actions, you might store the MCTS
                # probabilities for the state from self-play. For now, as an example,
                # we compute a dummy target policy only using the action at this step.
                n_actions: int = 3  # adjust as needed
                target_policy: np.ndarray = np.zeros(n_actions, dtype=np.float32)
                action = traj.actions[i]
                target_policy[action - 1] = 1.0  # one-hot target (replace with actual MCTS counts)
                self.data.append((state, target_policy, final_reward))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.data[idx]


def train_model(trajectories: Sequence[Any], num_epochs: int = 10, batch_size: int = 32) -> PVNetwork_Count21_TF:
    dataset_obj: TrajectoryDataset_Count21 = TrajectoryDataset_Count21(trajectories)

    # Generator yielding (state, label, target_value).
    # For policy loss, we use the class label (i.e. argmax of target_policy).
    def gen() -> Any:
        for state, target_policy, target_value in dataset_obj.data:
            label: int = int(np.argmax(target_policy))
            yield state, label, target_value

    state_dim: int = dataset_obj.data[0][0].shape[0]
    num_actions: int = dataset_obj.data[0][1].shape[0]
    num_players: int = dataset_obj.data[0][2].shape[0]

    # Output: (state, action, target_value)
    output_signature = (
        tf.TensorSpec(shape=(state_dim,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(num_players,), dtype=tf.float32),
    )
    tf_dataset: tf.data.Dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    tf_dataset = tf_dataset.shuffle(buffer_size=1000).batch(batch_size)

    model: PVNetwork_Count21_TF = PVNetwork_Count21_TF(
        state_dim=state_dim, num_actions=num_actions, num_players=num_players
    )

    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(1e-3)
    # For policy loss, we use SparseCategoricalCrossentropy (from_logits=True as we output raw logits).
    policy_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    value_loss_fn = tf.keras.losses.MeanSquaredError()

    for epoch in range(num_epochs):
        epoch_loss: float = 0.0
        steps: int = 0
        for states, labels, target_values in tf_dataset:
            with tf.GradientTape() as tape:
                policy_logits, value_pred = model(states, training=True)
                policy_loss: tf.Tensor = policy_loss_fn(labels, policy_logits)
                value_loss: tf.Tensor = value_loss_fn(target_values, value_pred)
                loss: tf.Tensor = policy_loss + value_loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()
            steps += 1
        print(f"Epoch {epoch + 1}: Average Loss = {epoch_loss / steps:.4f}")

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
    trajectories: Sequence[GameTrajectory[Any, Any]] = archiver.read_items(args.input, GameTrajectory)

    model = train_model(trajectories, num_epochs=args.num_epochs)
    model.save_weights(args.output)
    print(f"Model training complete. Weights saved to {args.output}")


if __name__ == "__main__":
    main()
