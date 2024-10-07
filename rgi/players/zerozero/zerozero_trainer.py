import functools
import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
from typing import Any, Tuple, List
import numpy as np
from tqdm import tqdm
import jax.tree_util as jtu
import pickle

from rgi.core.trajectory import load_trajectories, EncodedTrajectory
from rgi.players.zerozero.zerozero_model import ZeroZeroModel, zerozero_loss


class ZeroZeroTrainer:
    def __init__(self, model: ZeroZeroModel, learning_rate: float = 1e-4):
        self.model = model
        self.optimizer = optax.adam(learning_rate)
        self.state = None

    def create_train_state(self, rng: jax.random.PRNGKey) -> train_state.TrainState:
        params = self.model.init(rng, None, None)
        return train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.optimizer)

    @functools.partial(jax.jit, static_argnums=0)
    def train_step(self, state: train_state.TrainState, batch: Tuple[Any, ...]) -> Tuple[train_state.TrainState, dict]:
        def loss_fn(params):
            state_input, action, next_state, reward, policy_target = batch
            reward = jnp.asarray(reward)  # Ensure reward is a jax.Array
            loss, loss_dict = zerozero_loss(self.model, params, state_input, action, next_state, reward, policy_target)
            return loss, loss_dict

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, loss_dict), grads = grad_fn(state.params)
        return state.apply_gradients(grads=grads), loss_dict

    def create_batches(self, trajectories: List[EncodedTrajectory], batch_size: int):
        states, actions, next_states, rewards, policy_targets = [], [], [], [], []

        for trajectory in trajectories:
            for i in range(trajectory.length - 1):
                states.append(trajectory.states[i])
                actions.append(trajectory.actions[i])
                next_states.append(trajectory.states[i + 1])
                rewards.append(trajectory.state_rewards[i])
                policy_targets.append(
                    jax.nn.one_hot(trajectory.actions[i], num_classes=len(self.model.possible_actions))
                )

        dataset = list(zip(states, actions, next_states, rewards, policy_targets))
        np.random.shuffle(dataset)

        for i in range(0, len(dataset), batch_size):
            yield tuple(map(np.array, zip(*dataset[i : i + batch_size])))

    def train(self, trajectories_file: str, num_epochs: int, batch_size: int):
        trajectories = load_trajectories(trajectories_file)

        if self.state is None:
            rng = jax.random.PRNGKey(0)
            self.state = self.create_train_state(rng)

        for epoch in range(num_epochs):
            epoch_losses = []
            batches = self.create_batches(trajectories, batch_size)

            with tqdm(total=len(trajectories), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
                for batch in batches:
                    self.state, loss_dict = self.train_step(self.state, batch)
                    epoch_losses.append(loss_dict["total_loss"])
                    pbar.update(batch_size)
                    pbar.set_postfix({"loss": np.mean(epoch_losses)})

            print(f"Epoch {epoch + 1} - Average Loss: {np.mean(epoch_losses):.4f}")

    def save_checkpoint(self, checkpoint_dir: str):
        if self.state is None:
            raise ValueError("No state to save. Please train the model first.")
        checkpoints.save_checkpoint(checkpoint_dir, self.state, step=self.state.step, keep=3)

    def load_checkpoint(self, checkpoint_dir: str):
        if self.state is None:
            # We need a state of the correct type to restore from a checkpoint.
            rng = jax.random.PRNGKey(0)
            self.state = self.create_train_state(rng)
        self.state = checkpoints.restore_checkpoint(checkpoint_dir, target=self.state)
        if self.state is None:
            raise ValueError("No checkpoint found.")
