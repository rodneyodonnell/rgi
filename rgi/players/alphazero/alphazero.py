from typing import Generic, Optional, Literal, Sequence
from abc import ABC, abstractmethod
from typing_extensions import override
import numpy as np

from rgi.core.base import Player, TGame, TGameState, TAction

TPlayerState = Literal[None]


# Policy–Value Network interface.
# For multi‐player support, the network returns a value vector [v1, v2, ...] (one per player).
class PolicyValueNetwork(ABC, Generic[TGame, TGameState, TAction]):
    @abstractmethod
    def predict(
        self, game: TGame, state: TGameState, actions: Sequence[TAction]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given a game, state and list of legal actions, return a tuple (policy_logits, value)
          - policy_logits: 1D NumPy array of logits corresponding to each action.
          - value: NumPy array of predicted values, one per player. These should be in range (-1, 1).
        """


# A simple dummy implementation useful for testing MCTS.
class DummyPolicyValueNetwork(PolicyValueNetwork[TGame, TGameState, TAction]):
    @override
    def predict(
        self, game: TGame, state: TGameState, actions: Sequence[TAction]
    ) -> tuple[np.ndarray, np.ndarray]:
        num_actions = len(actions)
        policy_logits = np.log(np.ones(num_actions, dtype=np.float32) / num_actions)
        n_players = game.num_players(state)
        value = np.zeros(n_players, dtype=np.float32)
        return policy_logits, value


# AlphaZeroPlayer uses MCTS based on the policy–value network.
class AlphaZeroPlayer(
    Player[TGameState, TPlayerState, TAction], Generic[TGame, TGameState, TAction]
):
    def __init__(
        self, game: TGame, network: PolicyValueNetwork[TGame, TGameState, TAction]
    ) -> None:
        self.game = game
        self.network = network

    @override
    def select_action(
        self, game_state: TGameState, legal_actions: Sequence[TAction]
    ) -> TAction:
        mcts = MCTS(self.game, self.network, c_puct=1.0, num_simulations=50)
        action_visits = mcts.search(game_state)
        # Choose the action with the highest visit count.
        best_action = max(action_visits, key=action_visits.get)
        return best_action


# MCTS Node now stores a vector of total values.
class MCTSNode(Generic[TGame, TGameState, TAction]):
    def __init__(
        self,
        state: TGameState,
        n_players: int,
        parent: Optional["MCTSNode[TGame, TGameState, TAction]"] = None,
    ) -> None:
        self.state: TGameState = state
        self.parent: Optional["MCTSNode[TGame, TGameState, TAction]"] = parent
        self.legal_actions: Sequence[TAction] | None = None
        self.children: Sequence["MCTSNode[TGame, TGameState, TAction]"] | None = None
        self.visit_count: int = 0
        self.total_value = np.zeros(n_players, dtype=np.float32)
        self.prior: float = 0.0

    def q_value(self) -> np.ndarray:
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
        self.game = game
        self.network = network
        self.c_puct = c_puct  # Exploration constant.
        self.num_simulations = num_simulations
        self.n_players = game.num_players(game.initial_state())
        self.noise_alpha = noise_alpha
        self.noise_epsilon = noise_epsilon

    def search(self, root_state: TGameState) -> dict[TAction, int]:
        root: MCTSNode[TGame, TGameState, TAction] = MCTSNode(
            root_state, self.n_players
        )
        for _ in range(self.num_simulations):
            self._simulate(root)

        assert root.legal_actions is not None
        assert root.children is not None
        action_visits = {
            action: child.visit_count
            for action, child in zip(root.legal_actions, root.children)
        }
        return action_visits

    def _simulate(self, node: MCTSNode[TGame, TGameState, TAction]) -> np.ndarray:
        # Terminal state check.
        if self.game.is_terminal(node.state):
            # Expect reward to be a vector (absolute score per player).
            rewards = self.game.reward_array(node.state)
            return rewards

        legal_actions, children = node.legal_actions, node.children
        # Expansion: if node is a leaf (no children), expand using the network.
        if children is None or legal_actions is None:
            node.legal_actions = legal_actions = self.game.legal_actions(node.state)
            node.children = children = []

            policy_logits, action_values = self.network.predict(
                self.game, node.state, legal_actions
            )
            policy = self.softmax(policy_logits)

            if node.parent is None:
                noise = np.random.dirichlet([self.noise_alpha] * len(legal_actions))
                policy = (1 - self.noise_epsilon) * policy + self.noise_epsilon * noise

            # Create a new child for each legal action, using the corresponding probability.
            for index, action in enumerate(legal_actions):
                child_state = self.game.next_state(node.state, action)
                child_node = MCTSNode(child_state, self.n_players, parent=node)
                child_node.prior = policy[index]  # type: ignore
                children.append(child_node)
            return action_values

        # Selection: choose the child with highest UCB based on current player's value.
        curr_player = self.game.current_player_id(node.state)
        curr_index = curr_player - 1  # Adjust for 0-indexing.
        total_visits = sum(child.visit_count for child in children)
        best_score = -float("inf")
        best_child: Optional[MCTSNode[TGame, TGameState, TAction]] = None
        for action, child in zip(legal_actions, children):
            # ucb is the upper confidence bound of value for the action.
            # c_puct is a constant that controls the exploration-exploitation trade-off.
            ucb = child.q_value()[curr_index] + self.c_puct * child.prior * np.sqrt(
                total_visits
            ) / (1 + child.visit_count)
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

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
