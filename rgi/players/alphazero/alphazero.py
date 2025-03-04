import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, Optional, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from rgi.core.base import ActionResult, Player, TAction, TGame, TGameState

@dataclasses.dataclass
class MCTSData(Generic[TAction]):
    """Data collected during MCTS search for a single state.

    Args:
        policy_counts: Visit counts in MCTS, indexed by legal actions.
        prior_probabilities: Network's prior probabilities for each action, indexed by legal actions.
        value_estimate: Network's value estimate for each player, indexed by player_id.
        legal_actions: Sequence of legal actions for this state
    """

    policy_counts: dict[TAction, int]
    value_estimate: NDArray[np.float32]
    legal_actions: Sequence[TAction]

    root_prior_policies: NDArray[np.float32]
    root_prior_values: NDArray[np.float32]


TPlayerState = Literal[None]
TPlayerData = MCTSData[TAction]
TFloat = TypeVar("TFloat", bound=np.floating[Any])


# Policy–Value Network interface.
# For multi‐player support, the network returns a value vector [v1, v2, ...] (one per player).
class PolicyValueNetwork(ABC, Generic[TGame, TGameState, TAction]):
    @abstractmethod
    def predict(
        self, game: TGame, state: TGameState, legal_actions: Sequence[TAction]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Given a game, state and list of legal actions, return a tuple (policy_logits, value)
          - policy_logits: 1D NumPy array of logits corresponding to each action, indexed by legal action.
          - value: NumPy array of predicted values, one per player, indexed by player_id. These should be in range (-1, 1).
        """


# A simple dummy implementation useful for testing MCTS.
class DummyPolicyValueNetwork(PolicyValueNetwork[TGame, TGameState, TAction]):
    @override
    def predict(
        self, game: TGame, state: TGameState, legal_actions: Sequence[TAction]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        num_actions = len(legal_actions)
        policy_logits = np.log(np.ones(num_actions, dtype=np.float32) / num_actions)
        n_players = game.num_players(state)
        value = np.zeros(n_players, dtype=np.float32)
        return policy_logits, value


# AlphaZeroPlayer uses MCTS based on the policy–value network.
class AlphaZeroPlayer(Player[TGameState, TPlayerState, TAction, TPlayerData]):
    def __init__(
        self, game: TGame, network: PolicyValueNetwork[TGame, TGameState, TAction], *, num_simulations: int = 50
    ) -> None:
        self.game = game
        self.network = network
        self.num_simulations = num_simulations

    @override
    def select_action(
        self, game_state: TGameState, legal_actions: Sequence[TAction]
    ) -> ActionResult[TAction, TPlayerData]:
        mcts = MCTS(self.game, self.network, c_puct=1.0, num_simulations=self.num_simulations)

        # Run MCTS search
        mcts_data = mcts.search(game_state)
        
        # Validate that passed in and calculated legal actions are the same.
        assert mcts_data.legal_actions == legal_actions

        # Choose the action with the highest visit count
        best_action_index = np.argmax(mcts_data.policy_counts)
        best_action = mcts_data.legal_actions[best_action_index]

        return ActionResult(best_action, mcts_data)

    def softmax(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        e_x = np.exp(x - np.max(x))
        return np.array(e_x / e_x.sum(), dtype=np.float32)


class MCTSNode(Generic[TGame, TGameState, TAction]):
    def __init__(
        self,
        state: TGameState,
        n_players: int,
        parent: Optional["MCTSNode[TGame, TGameState, TAction]"] = None,
    ) -> None:
        self.state: TGameState = state
        self.parent: Optional["MCTSNode[TGame, TGameState, TAction]"] = parent
        self.legal_actions: Optional[Sequence[TAction]] = None
        self.children: Optional[Sequence["MCTSNode[TGame, TGameState, TAction]"]] = None
        self.visit_count: int = 0
        self.total_value: NDArray[np.float32] = np.zeros(n_players, dtype=np.float32)
        self.prior: float = 0.0

    def q_value(self) -> NDArray[np.float32]:
        # Return average value vector. If unvisited, return zeros.
        if self.visit_count == 0:
            return np.zeros_like(self.total_value)
        return self.total_value / self.visit_count


class MCTS(Generic[TGame, TGameState, TAction]):
    def __init__(
        self,
        game: TGame,
        network: PolicyValueNetwork[TGame, TGameState, TAction],
        c_puct: float = 1.0,
        num_simulations: int = 50,
        noise_alpha: float = 0.03,
        noise_epsilon: float = 0.25,
    ) -> None:
        assert num_simulations > 0
        self.game = game
        self.network = network
        self.c_puct = c_puct  # Exploration constant.
        self.num_simulations = num_simulations
        self.n_players = game.num_players(game.initial_state())
        self.noise_alpha = noise_alpha
        self.noise_epsilon = noise_epsilon
        self.root_prior_policies: Optional[NDArray[np.float32]] = None
        self.root_prior_values: Optional[NDArray[np.float32]] = None

    def search(self, root_state: TGameState) -> MCTSData[TAction]:
        root: MCTSNode[TGame, TGameState, TAction] = MCTSNode(root_state, self.n_players)
        for _ in range(self.num_simulations):
            self._simulate(root)

        assert root.legal_actions is not None
        assert root.children is not None
        action_visits = np.array([child.visit_count for child in root.children], dtype=np.int32)
        mcts_data: MCTSData[TAction] = MCTSData(
            policy_counts=action_visits,
            root_prior_policies=self.root_prior_policies,
            root_prior_values=self.root_prior_values,
            value_estimate=root.total_value / root.visit_count,
            legal_actions=root.legal_actions,
        )
        return mcts_data

    def _simulate(self, node: MCTSNode[TGame, TGameState, TAction]) -> NDArray[np.float32]:
        # Terminal state check.
        if self.game.is_terminal(node.state):
            # Expect reward to be a vector (absolute score per player).
            rewards = np.array(self.game.reward_array(node.state), dtype=np.float32)
            return rewards

        legal_actions, children = node.legal_actions, node.children
        # Expansion: if node is a leaf (no children), expand using the network.
        if children is None or legal_actions is None:
            node.legal_actions = legal_actions = self.game.legal_actions(node.state)
            node.children = children = []

            policy_logits, action_values = self.network.predict(self.game, node.state, legal_actions)
            policy = self.softmax(policy_logits)

            if node.parent is None:
                # Record extra data when expanding root node.
                self.root_prior_policies = self.softmax(policy_logits)
                self.root_prior_values = action_values
                # Add some noise to root node to make game non deterministic.
                noise = np.random.dirichlet([self.noise_alpha] * len(legal_actions))
                policy = (1 - self.noise_epsilon) * policy + self.noise_epsilon * noise

            # Create a new child for each legal action, using the corresponding probability.
            for index, action in enumerate(legal_actions):
                child_state = self.game.next_state(node.state, action)
                child_node = MCTSNode(child_state, self.n_players, parent=node)
                child_node.prior = float(policy[index])
                children.append(child_node)
            
            return action_values

        # Selection: choose the child with highest UCB (upper confidence bound) based on current player's value.
        curr_player = self.game.current_player_id(node.state)
        curr_index = curr_player - 1  # Adjust for 0-indexing.
        total_visits = sum(child.visit_count for child in children)
        best_score = -float("inf")
        best_child: Optional[MCTSNode[TGame, TGameState, TAction]] = None
        for action, child in zip(legal_actions, children):
            # ucb is the upper confidence bound of value for the action.
            # c_puct is a constant that controls the exploration-exploitation trade-off.
            ucb = child.q_value()[curr_index] + self.c_puct * child.prior * np.sqrt(total_visits) / (
                1 + child.visit_count
            )
            if ucb > best_score:
                best_score = ucb
                best_child = child
        if best_child is None:
            raise ValueError("No valid child found during selection in MCTS.")

        # Recursively simulate from the selected child.
        action_values = self._simulate(best_child)
        best_child.visit_count += 1
        best_child.total_value += action_values

        return action_values

    def softmax(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        e_x = np.exp(x - np.max(x))
        return np.array(e_x / e_x.sum(), dtype=np.float32)
