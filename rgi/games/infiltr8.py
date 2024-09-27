# rgi/games/infiltr8.py
"""
## Setup
1. Shuffle the deck and remove one card face down (unseen).
2. Deal one card to each player.

## Gameplay
1. On your turn:
   a. Draw one card from the deck.
   b. Choose one of your two cards to play, applying its effect.
   c. Discard the played card face up in front of you.

2. Card effects are resolved immediately after playing.

3. If a player is eliminated, they're out for the rest of the round.

4. Play continues clockwise until either:
   a. The deck is empty, or
   b. All but one player are eliminated.

## Winning a Round
- If only one player remains, they win the round.
- If the deck is empty, the player with the highest value card wins the round.
- In case of a tie, players share the win.

## Game End
- Play multiple rounds (typically first to 7 points).
- Win a round to score a point.

## Key Rules
- Players must always tell the truth about their cards.
- If a player has no legal play, they must play a card anyway.
- Eliminated players' discards remain in play for card counting.
"""

from dataclasses import dataclass, replace
from typing import Optional, Any, Iterator
import random
from enum import Enum
from immutables import Map
from typing_extensions import override

from rgi.core.base import Game, GameSerializer

TPlayerId = int


class ActionType(Enum):
    DRAW = 1
    PLAY = 2


# fmt: off
class CardEffect(Enum):
    GUESS = 1          # Player guesses a card value in another player's hand
    PEEK = 2           # Player looks at another player's hand without revealing it
    COMPARE = 3        # Player compares their card with another player's card
    PROTECT = 4        # Player becomes immune to effects until their next turn
    FORCE_DISCARD = 5  # Player forces another player to discard their card
    SWAP = 6           # Player swaps their card with another player's card
    CONDITIONAL_DISCARD = 7  # Player must discard if holding certain cards
    LOSE = 8           # Player is immediately eliminated if they discard this card
# fmt: on


class TurnPhase(Enum):
    DRAW = 1
    PLAY = 2


@dataclass(frozen=True)
class Card:
    name: str
    effect: CardEffect


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    card: Optional[Card]
    player_id: Optional[TPlayerId]
    value: Optional[int]


@dataclass(frozen=True)
class PlayerState:
    hand: tuple[Card, ...]
    is_protected: bool
    is_out: bool


@dataclass(frozen=True)
class PendingAction:
    player_id: TPlayerId
    effect: CardEffect


@dataclass(frozen=True)
class Infiltr8State:
    deck: tuple[Card, ...]
    players: Map[TPlayerId, PlayerState]
    discard_pile: tuple[Card, ...]
    current_player_turn: TPlayerId  # Player ID of the player whose turn it is
    pending_action: Optional[PendingAction]
    turn_phase: TurnPhase


GUESS_CARD = Card(name="Guess", effect=CardEffect.GUESS)
PEEK_CARD = Card(name="Peek", effect=CardEffect.PEEK)
COMPARE_CARD = Card(name="Compare", effect=CardEffect.COMPARE)
PROTECT_CARD = Card(name="Protect", effect=CardEffect.PROTECT)
FORCE_DISCARD_CARD = Card(name="Force Discard", effect=CardEffect.FORCE_DISCARD)
SWAP_CARD = Card(name="Swap", effect=CardEffect.SWAP)
CONDITIONAL_DISCARD_CARD = Card(name="Conditional Discard", effect=CardEffect.CONDITIONAL_DISCARD)
LOSE_CARD = Card(name="Lose", effect=CardEffect.LOSE)

UNIQUE_CARDS = [
    GUESS_CARD,
    PEEK_CARD,
    COMPARE_CARD,
    PROTECT_CARD,
    FORCE_DISCARD_CARD,
    SWAP_CARD,
    CONDITIONAL_DISCARD_CARD,
    LOSE_CARD,
]


class Infiltr8Game(Game[Infiltr8State, TPlayerId, Action]):

    def __init__(self, num_players: int = 4):
        if num_players < 2 or num_players > 4:
            raise ValueError("Number of players must be between 2 and 4")
        self.num_players = num_players
        self.full_deck = (
            [GUESS_CARD] * 5
            + [PEEK_CARD] * 2
            + [COMPARE_CARD] * 2
            + [PROTECT_CARD] * 2
            + [FORCE_DISCARD_CARD] * 2
            + [SWAP_CARD, CONDITIONAL_DISCARD_CARD, LOSE_CARD]
        )

    @override
    def initial_state(self, random_seed: Optional[int] = None) -> Infiltr8State:
        rng = random.Random(random_seed)
        deck = list(self.full_deck)
        rng.shuffle(deck)
        _removed_card = deck.pop()
        players = {}
        for i in range(1, self.num_players + 1):
            players[i] = PlayerState(hand=(deck.pop(),), is_protected=False, is_out=False)
        return Infiltr8State(
            deck=tuple(deck),
            players=Map(players),
            discard_pile=(),
            current_player_turn=1,
            pending_action=None,
            turn_phase=TurnPhase.DRAW,
        )

    @override
    def current_player_id(self, state: Infiltr8State) -> TPlayerId:
        if state.pending_action:
            return state.pending_action.player_id  # Return the ID of the player with the pending action
        return state.current_player_turn

    @override
    def all_player_ids(self, state: Infiltr8State) -> list[TPlayerId]:
        return list(range(1, self.num_players + 1))

    def _legal_play_actions_iter(self, state: Infiltr8State) -> Iterator[Action]:
        turn_player = state.current_player_turn
        player_state = state.players[turn_player]

        # If we have a CONDITIONAL_DISCARD card, we must play it if we also have a SWAP or FORCE_DISCARD card
        if any(card.effect == CardEffect.CONDITIONAL_DISCARD for card in player_state.hand):
            if any(card.effect in (CardEffect.SWAP, CardEffect.FORCE_DISCARD) for card in player_state.hand):
                yield Action(action_type=ActionType.PLAY, card=CONDITIONAL_DISCARD_CARD, player_id=None, value=None)
                return

        for action_card in player_state.hand:
            if action_card.effect == CardEffect.GUESS:
                for other_player in self.all_player_ids(state):
                    if other_player != turn_player and not state.players[other_player].is_protected:
                        for value_card in UNIQUE_CARDS:
                            yield Action(
                                action_type=ActionType.PLAY,
                                card=action_card,
                                player_id=other_player,
                                value=value_card.effect.value,
                            )
            elif action_card.effect in (CardEffect.PEEK, CardEffect.COMPARE, CardEffect.SWAP):
                for other_player in self.all_player_ids(state):
                    if other_player != turn_player and not state.players[other_player].is_protected:
                        yield Action(action_type=ActionType.PLAY, card=action_card, player_id=other_player, value=None)
            elif action_card.effect in (CardEffect.FORCE_DISCARD,):
                for target_player in self.all_player_ids(state):
                    yield Action(action_type=ActionType.PLAY, card=action_card, player_id=target_player, value=None)
            elif action_card.effect in (CardEffect.PROTECT, CardEffect.CONDITIONAL_DISCARD, CardEffect.LOSE):
                yield Action(action_type=ActionType.PLAY, card=action_card, player_id=None, value=None)
            else:
                raise ValueError(f"Unknown card effect: {action_card}")

    @override
    def legal_actions(self, state: Infiltr8State) -> list[Action]:

        turn_player = state.current_player_turn
        player_state = state.players[turn_player]

        if self.is_terminal(state):
            raise ValueError("No legal actions in terminal state")
        if player_state.is_out:
            raise ValueError("No legal actions in for out players")

        if state.pending_action:
            raise NotImplementedError(f"Pending action not implemented yet for {state.pending_action}")

        if state.turn_phase == TurnPhase.DRAW:
            return [Action(action_type=ActionType.DRAW, card=None, player_id=None, value=None)]

        return list(self._legal_play_actions_iter(state))

    @override
    def next_state(self, state: Infiltr8State, action: Action) -> Infiltr8State:
        if self.is_terminal(state):
            raise ValueError("No legal actions in terminal state")
        if state.players[state.current_player_turn].is_out:
            raise ValueError("No legal actions for out players")

        if action.action_type == ActionType.DRAW:
            return self._handle_draw_action(state, action)
        if action.action_type == ActionType.PLAY:
            return self._handle_play_action(state, action)
        raise ValueError(f"Unknown action type: {action.action_type}")

    def _handle_draw_action(self, state: Infiltr8State, _action: Action) -> Infiltr8State:
        turn_player = state.current_player_turn
        player_state = state.players[turn_player]
        new_card, new_deck = state.deck[0], state.deck[1:]
        new_player_state = replace(player_state, hand=player_state.hand + (new_card,))
        return replace(
            state,
            players=state.players.set(turn_player, new_player_state),
            deck=new_deck,
            turn_phase=TurnPhase.PLAY,
        )

    def _handle_play_action(self, state: Infiltr8State, action: Action) -> Infiltr8State:
        turn_player = state.current_player_turn
        player_state = state.players[turn_player]
        played_card = action.card
        if played_card is None:
            raise ValueError("No card specified for Play action")

        new_hand = tuple(card for card in player_state.hand if card != played_card) or (player_state.hand[0],)
        new_player_state = replace(player_state, hand=new_hand, is_protected=False)
        new_players = state.players.set(turn_player, new_player_state)
        new_discard_pile = state.discard_pile + (played_card,)

        new_state = replace(state, players=new_players, discard_pile=new_discard_pile)
        new_state = self._apply_card_effect(new_state, action)

        next_player = self._get_next_player(new_state)
        return replace(
            new_state,
            current_player_turn=next_player,
            turn_phase=TurnPhase.DRAW,
        )

    def _apply_card_effect(self, state: Infiltr8State, action: Action) -> Infiltr8State:
        assert action.card is not None
        effect_handlers = {
            CardEffect.GUESS: self._handle_guess_effect,
            CardEffect.PEEK: self._handle_peek_effect,
            CardEffect.COMPARE: self._handle_compare_effect,
            CardEffect.PROTECT: self._handle_protect_effect,
            CardEffect.FORCE_DISCARD: self._handle_force_discard_effect,
            CardEffect.SWAP: self._handle_swap_effect,
            CardEffect.CONDITIONAL_DISCARD: self._handle_conditional_discard_effect,
            CardEffect.LOSE: self._handle_lose_effect,
        }

        handler = effect_handlers.get(action.card.effect)
        if not handler:
            raise ValueError(f"Unknown card effect: {action.card.effect}")
        return handler(state, action)

    def _handle_guess_effect(self, state: Infiltr8State, action: Action) -> Infiltr8State:
        if action.player_id is None:
            raise ValueError("No target player specified for GUESS action")
        target_player = state.players[action.player_id]
        if target_player.hand[0].effect.value == action.value:
            new_target_player = replace(target_player, is_out=True)
            new_players = state.players.set(action.player_id, new_target_player)
            return replace(state, players=new_players)
        return state

    def _handle_peek_effect(self, state: Infiltr8State, _action: Action) -> Infiltr8State:
        # PEEK effect doesn't change the game state
        return state

    def _handle_compare_effect(self, state: Infiltr8State, action: Action) -> Infiltr8State:
        if action.player_id is None:
            raise ValueError("No target player specified for COMPARE action")
        turn_player = state.current_player_turn
        player_card = state.players[turn_player].hand[0]
        target_player = state.players[action.player_id]
        target_card = target_player.hand[0]
        new_players = state.players
        if player_card.effect.value < target_card.effect.value:
            new_player_state = replace(state.players[turn_player], is_out=True)
            new_players = new_players.set(turn_player, new_player_state)
        elif player_card.effect.value > target_card.effect.value:
            new_target_player = replace(target_player, is_out=True)
            new_players = new_players.set(action.player_id, new_target_player)
        return replace(state, players=new_players)

    # TODO: We shoudl check for protection in legal_actions? and again in apply_action?
    def _handle_protect_effect(self, state: Infiltr8State, _action: Action) -> Infiltr8State:
        turn_player = state.current_player_turn
        new_player_state = replace(state.players[turn_player], is_protected=True)
        new_players = state.players.set(turn_player, new_player_state)
        return replace(state, players=new_players)

    def _handle_force_discard_effect(self, state: Infiltr8State, action: Action) -> Infiltr8State:
        if action.player_id is None:
            raise ValueError("No target player specified for FORCE_DISCARD action")
        target_player = state.players[action.player_id]
        if not target_player.is_protected:
            discarded_card = target_player.hand[0]
            new_hand, new_deck = (state.deck[0],), state.deck[1:]
            new_target_player = replace(target_player, hand=new_hand)
            new_players = state.players.set(action.player_id, new_target_player)
            new_discard_pile = state.discard_pile + (discarded_card,)
            return replace(state, players=new_players, discard_pile=new_discard_pile, deck=new_deck)
        return state

    def _handle_swap_effect(self, state: Infiltr8State, action: Action) -> Infiltr8State:
        if action.player_id is None:
            raise ValueError("No target player specified for SWAP action")
        turn_player = state.current_player_turn
        target_player = state.players[action.player_id]
        if not target_player.is_protected:
            player_card = state.players[turn_player].hand[0]
            target_card = target_player.hand[0]
            new_player_state = replace(state.players[turn_player], hand=(target_card,))
            new_target_player = replace(target_player, hand=(player_card,))
            new_players = state.players.set(turn_player, new_player_state).set(action.player_id, new_target_player)
            return replace(state, players=new_players)
        return state

    def _handle_conditional_discard_effect(self, state: Infiltr8State, _action: Action) -> Infiltr8State:
        turn_player = state.current_player_turn
        player_state = state.players[turn_player]
        if any(card.effect in (CardEffect.SWAP, CardEffect.FORCE_DISCARD) for card in player_state.hand):
            discarded_card = next(
                card for card in player_state.hand if card.effect in (CardEffect.SWAP, CardEffect.FORCE_DISCARD)
            )
            new_hand = tuple(card for card in player_state.hand if card != discarded_card)
            new_player_state = replace(player_state, hand=new_hand)
            new_players = state.players.set(turn_player, new_player_state)
            new_discard_pile = state.discard_pile + (discarded_card,)
            return replace(state, players=new_players, discard_pile=new_discard_pile)
        return state

    def _handle_lose_effect(self, state: Infiltr8State, _action: Action) -> Infiltr8State:
        turn_player = state.current_player_turn
        new_player_state = replace(state.players[turn_player], is_out=True)
        new_players = state.players.set(turn_player, new_player_state)
        return replace(state, players=new_players)

    def _get_next_player(self, state: Infiltr8State) -> TPlayerId:
        current_player = state.current_player_turn
        for i in range(1, self.num_players + 1):
            next_player = (current_player + i) % self.num_players
            if next_player == 0:
                next_player = self.num_players
            if not state.players[next_player].is_out:
                return next_player
        raise ValueError("No active players remaining")

    @override
    def is_terminal(self, state: Infiltr8State) -> bool:
        active_players = sum(1 for player in state.players.values() if not player.is_out)
        return active_players <= 1 or len(state.deck) == 0

    @override
    def reward(self, state: Infiltr8State, player_id: TPlayerId) -> float:
        if not self.is_terminal(state):
            return 0.0

        active_players = [pid for pid, player in state.players.items() if not player.is_out]

        if len(active_players) == 1:
            return 1.0 if player_id == active_players[0] else -1.0

        # If multiple players are active (deck is empty), highest card wins
        highest_value = max(player.hand[0].effect.value for player in state.players.values() if not player.is_out)
        winners = [
            pid
            for pid, player in state.players.items()
            if not player.is_out and player.hand[0].effect.value == highest_value
        ]

        if player_id in winners:
            return 1.0 if len(winners) == 1 else 0.5  # Full points for sole winner, half for tie
        return -1.0

    @override
    def pretty_str(self, state: Infiltr8State) -> str:
        output = []
        output.append(f"Current player: {state.current_player_turn}")
        output.append(f"Deck size: {len(state.deck)}")
        output.append(f"Discard pile: {', '.join(card.name for card in state.discard_pile[-3:])}")
        for player_id, player_state in state.players.items():
            status = "Protected" if player_state.is_protected else "Active"
            if player_state.is_out:
                status = "Out"
            output.append(f"Player {player_id}: {status}, Hand: {', '.join(card.name for card in player_state.hand)}")
        return "\n".join(output)


class Infiltr8Serializer(GameSerializer[Infiltr8Game, Infiltr8State, Action]):
    @override
    def serialize_state(self, game: Infiltr8Game, state: Infiltr8State) -> dict[str, Any]:
        return {
            "current_player": state.current_player_turn,
            "deck_size": len(state.deck),
            "discard_pile": [{"name": card.name, "value": card.effect.value} for card in state.discard_pile[-3:]],
            "players": {
                player_id: {
                    "is_protected": player_state.is_protected,
                    "is_out": player_state.is_out,
                    "hand_size": len(player_state.hand),
                }
                for player_id, player_state in state.players.items()
            },
            "legal_actions": [self.serialize_action(action) for action in game.legal_actions(state)],
        }

    def serialize_action(self, action: Action) -> dict[str, Any]:
        return {
            "action_type": action.action_type,
            "card": action.card,
            "player_id": action.player_id,
            "value": action.value,
        }

    @override
    def parse_action(self, game: Infiltr8Game, action_data: dict[str, Any]) -> Action:
        return Action(
            action_type=action_data["action_type"],
            card=action_data["card"],
            player_id=action_data["player_id"],
            value=action_data["value"],
        )
