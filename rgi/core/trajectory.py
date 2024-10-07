# rgi/core/trajectory.py

from dataclasses import dataclass
from typing import Generic, TypeVar, List
import jax
import jax.numpy as jnp
import glob
import numpy as np

from rgi.core.base import Game, TGameState, TAction, TPlayerId

T = TypeVar("T")


@dataclass
class Trajectory(Generic[TGameState, TAction, TPlayerId]):
    states: list[TGameState]
    actions: list[TAction]
    state_rewards: list[float]  # Incremental reward for each state/action pair
    player_ids: list[TPlayerId]  # Player who took the action
    final_rewards: list[float]  # Final reward for each player


@dataclass
class EncodedTrajectory:
    states: jnp.ndarray
    actions: jnp.ndarray
    state_rewards: jnp.ndarray
    player_ids: jnp.ndarray
    final_rewards: jnp.ndarray
    length: int  # Actual length of the trajectory


def encode_trajectory(
    game: Game, trajectory: Trajectory[TGameState, TAction, TPlayerId], game_serializer
) -> EncodedTrajectory:
    encoded_states = jnp.stack([game_serializer.state_to_jax_array(game, state) for state in trajectory.states])
    encoded_actions = jnp.stack([game_serializer.action_to_jax_array(game, action) for action in trajectory.actions])
    encoded_state_rewards = jnp.array(trajectory.state_rewards)
    encoded_player_ids = jnp.array(trajectory.player_ids)
    encoded_final_rewards = jnp.array(trajectory.final_rewards)
    return EncodedTrajectory(
        encoded_states,
        encoded_actions,
        encoded_state_rewards,
        encoded_player_ids,
        encoded_final_rewards,
        len(trajectory.states),
    )


def save_trajectories(trajectories: list[EncodedTrajectory], filename: str):
    data = {
        "states": [t.states for t in trajectories],
        "actions": [t.actions for t in trajectories],
        "state_rewards": [t.state_rewards for t in trajectories],
        "player_ids": [t.player_ids for t in trajectories],
        "final_rewards": [t.final_rewards for t in trajectories],
        "lengths": [t.length for t in trajectories],
    }
    jnp.save(filename, data, allow_pickle=True)


def load_trajectories(trajectories_glob: str) -> list[EncodedTrajectory]:
    return [
        EncodedTrajectory(
            states=data["states"][i],
            actions=data["actions"][i],
            state_rewards=data["state_rewards"][i],
            player_ids=data["player_ids"][i],
            final_rewards=data["final_rewards"][i],
            length=data["lengths"][i],
        )
        for file_path in glob.glob(trajectories_glob)
        for data in [jnp.load(file_path, allow_pickle=True).item()]
        for i in range(len(data["lengths"]))
    ]
