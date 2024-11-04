# rgi/core/trajectory.py

from dataclasses import dataclass
from typing import Generic, TypeVar, List
import torch
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
    states: torch.Tensor
    actions: torch.Tensor
    state_rewards: torch.Tensor
    player_ids: torch.Tensor
    final_rewards: torch.Tensor
    num_actions: int  # Total number of actions in the trajectory.
    num_players: int


def encode_trajectory(
    game: Game, trajectory: Trajectory[TGameState, TAction, TPlayerId], game_serializer
) -> EncodedTrajectory:
    encoded_states = torch.stack([game_serializer.state_to_tensor(game, state) for state in trajectory.states])
    encoded_actions = torch.stack([game_serializer.action_to_tensor(game, action) for action in trajectory.actions])
    encoded_state_rewards = torch.tensor(trajectory.state_rewards)
    encoded_player_ids = torch.tensor(trajectory.player_ids)
    encoded_final_rewards = torch.tensor(trajectory.final_rewards)
    return EncodedTrajectory(
        encoded_states,
        encoded_actions,
        encoded_state_rewards,
        encoded_player_ids,
        encoded_final_rewards,
        len(trajectory.actions),
        len(trajectory.final_rewards),
    )


def save_trajectories(trajectories: List[EncodedTrajectory], filename: str) -> None:
    trajectory_data = {
        "states": torch.cat([t.states for t in trajectories]),
        "actions": torch.cat([t.actions for t in trajectories]),
        "state_rewards": torch.cat([t.state_rewards for t in trajectories]),
        "player_ids": torch.cat([t.player_ids for t in trajectories]),
        "final_rewards": torch.cat([t.final_rewards for t in trajectories]),
        "num_actions": torch.tensor([t.actions.shape[0] for t in trajectories]),
        "num_players": torch.tensor([t.final_rewards.shape[0] for t in trajectories]),
    }
    torch.save(trajectory_data, filename)


def load_trajectories(filename_or_glob: str) -> List[EncodedTrajectory]:
    filenames = glob.glob(filename_or_glob)
    trajectories = []
    for filename in filenames:
        trajectory_data = torch.load(filename)
        state_idx = 0
        action_idx = 0
        reward_idx = 0
        for num_actions, num_players in zip(trajectory_data["num_actions"], trajectory_data["num_players"]):
            state_end_idx = state_idx + num_actions + 1  # +1 for initial state.
            action_end_idx = action_idx + num_actions
            reward_end_idx = reward_idx + num_players
            trajectories.append(
                EncodedTrajectory(
                    states=trajectory_data["states"][state_idx:state_end_idx],
                    actions=trajectory_data["actions"][action_idx:action_end_idx],
                    state_rewards=trajectory_data["state_rewards"][state_idx:state_end_idx],
                    player_ids=trajectory_data["player_ids"][action_idx:action_end_idx],
                    final_rewards=trajectory_data["final_rewards"][reward_idx:reward_end_idx],
                    num_actions=num_actions,
                    num_players=num_players,
                )
            )
            state_idx = state_end_idx
            action_idx = action_end_idx
            reward_idx = reward_end_idx
    return trajectories
