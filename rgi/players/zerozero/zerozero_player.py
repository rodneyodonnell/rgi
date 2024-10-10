from typing import Generic, Any
from typing_extensions import override
import jax
import jax.numpy as jnp
from rgi.core.base import Game, Player, TGameState, TPlayerState, TAction, GameSerializer
from rgi.players.zerozero.zerozero_model import ZeroZeroModel
import jax.numpy as jnp
from rgi.players.zerozero.zerozero_model import StateEmbedder, ActionEmbedder


class ZeroZeroPlayer(Generic[TGameState, TPlayerState, TAction], Player[TGameState, TPlayerState, TAction]):
    def __init__(
        self,
        zerozero_model: ZeroZeroModel[TGameState, TPlayerState, TAction],
        zerozero_model_params: dict,
        game: Game[TGameState, TPlayerState, TAction],
        serializer: GameSerializer[TGameState, TPlayerState, TAction],
        temperature: float = 1.0,
        rng_key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ):
        self.zerozero_model = zerozero_model
        self.zerozero_model_params = zerozero_model_params
        self.temperature = temperature
        self.rng_key = rng_key
        self.game = game
        self.serializer = serializer

    @override
    def select_action(self, game_state: TGameState, legal_actions: list[TAction]) -> TAction:
        self.rng_key, subkey = jax.random.split(self.rng_key)

        # Get policy logits from the model
        encoded_game_state = self.serializer.state_to_jax_array(self.game, game_state)
        encoded_game_state_batch = jnp.expand_dims(encoded_game_state, axis=0)
        _next_state_embedding, _next_state_reward, policy_embedding = self.zerozero_model.apply(self.zerozero_model_params, encoded_game_state_batch, action=None)  # type: ignore
        action_logits = self.zerozero_model.apply(
            self.zerozero_model_params,
            method=self.zerozero_model.compute_action_logits,
            policy_embedding=policy_embedding,
        )

        # Mask illegal actions
        mask = jnp.array([1.0 if a in legal_actions else 0.0 for a in self.zerozero_model.possible_actions])
        masked_logits = action_logits * mask - 1e9 * (1 - mask)

        # Apply temperature
        scaled_logits = masked_logits / self.temperature

        # Compute probabilities
        action_probs_batch = jax.nn.softmax(scaled_logits)
        action_probs = action_probs_batch[0]

        # Sample action based on probabilities
        action_index = jax.random.choice(
            subkey,
            len(self.zerozero_model.possible_actions),
            p=action_probs,
        )

        return self.zerozero_model.possible_actions[action_index]

    @override
    def update_state(self, game_state: TGameState, action: TAction) -> None:
        # This method can be used to update any internal state if needed
        pass
