import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax


# Simple dataset generation
def generate_data() -> None:
    X = np.random.rand(10000, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(10000) * 0.1
    data = np.hstack([X, y[:, None]])
    np.save("data.npy", data)
    print("Generated data and saved to data.npy")


# Define a simple model
class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(features=1)(x)
        return x


# Training function
def train_model() -> None:
    data = np.load("data.npy")
    X = data[:, :1]
    y = data[:, 1:]

    batch_size = 32
    num_epochs = 5
    steps_per_epoch = X.shape[0] // batch_size

    model = SimpleModel()

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones([1, 1]))["params"]

    tx = optax.adam(learning_rate=0.001)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    # Use absolute path for checkpoint directory
    checkpoint_dir = os.path.abspath("./checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Restore the latest checkpoint if it exists
    latest_checkpoint = checkpoints.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        state = checkpoints.restore_checkpoint(latest_checkpoint, state)
        print(f"Restored checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found, starting from scratch.")

    @jax.jit
    def train_step(state, x, y):
        def loss_fn(params):
            preds = state.apply_fn({"params": params}, x)
            loss = jnp.mean((preds - y) ** 2)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    for epoch in range(num_epochs):
        perm = np.random.permutation(X.shape[0])
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        for step in range(steps_per_epoch):
            x_batch = X_shuffled[step * batch_size : (step + 1) * batch_size]
            y_batch = y_shuffled[step * batch_size : (step + 1) * batch_size]
            state, loss = train_step(state, x_batch, y_batch)
            if step % 10 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}")
        checkpoints.save_checkpoint(checkpoint_path, state, step, keep=3, overwrite=True)
        print(f"Saved checkpoint at epoch {epoch+1}")

    print("Training complete")


# Prediction function
def predict() -> None:
    data = np.load("data.npy")
    X = data[:, :1]
    y_true = data[:, 1:]

    model = SimpleModel()

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones([1, 1]))["params"]

    tx = optax.adam(learning_rate=0.001)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Use absolute path for checkpoint directory
    checkpoint_dir = os.path.abspath("./checkpoints")

    # Find the latest checkpoint
    latest_checkpoint = checkpoints.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is None:
        raise ValueError("No checkpoint found. Please train the model first.")

    # Restore the latest checkpoint
    state = checkpoints.restore_checkpoint(latest_checkpoint, state)
    print(f"Restored checkpoint: {latest_checkpoint}")

    preds = model.apply({"params": state.params}, X)
    mse = np.mean((preds - y_true) ** 2)
    print(f"Mean squared error on the dataset: {mse}")

    for i in range(5):
        print(f"Input: {X[i][0]:.4f}, Predicted: {preds[i][0]:.4f}, True: {y_true[i][0]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["gen_data", "train", "predict"], help="Mode to run")
    args = parser.parse_args()

    if args.mode == "gen_data":
        generate_data()
    elif args.mode == "train":
        train_model()
    elif args.mode == "predict":
        predict()
    else:
        print("Invalid mode")


if __name__ == "__main__":
    main()
