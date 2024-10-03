# rgi/tests/games/test_infiltr8.py

from dataclasses import replace
from typing import Any
import pytest
from rgi.games.infiltr8 import (
    Infiltr8Game,
    Infiltr8State,
    Card,
    Action,
    ActionType,
    TurnPhase,
    Infiltr8Serializer,
    CARD_NAMES,
)

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def game() -> Infiltr8Game:
    return Infiltr8Game(num_players=2)


@pytest.fixture
def game_4p() -> Infiltr8Game:
    return Infiltr8Game(num_players=4)


@pytest.fixture
def serializer() -> Infiltr8Serializer:
    return Infiltr8Serializer()


DRAW_ACTION = Action(ActionType.DRAW, card=None, player_id=None, guess_card=None)


def stack_deck(state: Infiltr8State, card: Card) -> Infiltr8State:
    """Add a card on top of the deck."""
    return replace(state, deck=(card,) + state.deck)


def set_hand(state: Infiltr8State, player_id: int, hand: tuple[Card, ...]) -> Infiltr8State:
    updated_player = replace(state.players[player_id], hand=hand)
    updated_players = state.players.set(player_id, updated_player)
    return replace(state, players=updated_players)


def test_initial_state(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=123)
    assert len(state.players) == 2
    assert len(state.deck) == 16 - len(state.players) - 1
    for player_state in state.players.values():
        assert len(player_state.hand) == 1
        assert not player_state.is_protected
        assert not player_state.is_out
    assert state.current_player_turn == 1
    assert game.all_player_ids(state) == [1, 2]
    assert game.is_terminal(state) is False
    assert game.reward(state, 1) == 0
    assert game.reward(state, 2) == 0


def test_initial_state_4p(game_4p: Infiltr8Game) -> None:
    state = game_4p.initial_state(random_seed=123)
    assert len(state.players) == 4
    assert len(state.deck) == 16 - len(state.players) - 1
    for player_state in state.players.values():
        assert len(player_state.hand) == 1
        assert not player_state.is_protected
        assert not player_state.is_out
    assert state.current_player_turn == 1
    assert game_4p.all_player_ids(state) == [1, 2, 3, 4]


def test_initial_actions(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=123)
    assert game.legal_actions(state) == [DRAW_ACTION]


def test_first_draw_action(game: Infiltr8Game) -> None:
    old_state = game.initial_state(random_seed=123)
    old_hand = old_state.players[1].hand
    old_deck = old_state.deck
    assert len(old_hand) == 1
    assert len(old_deck) == 13

    new_state = game.next_state(old_state, DRAW_ACTION)
    assert new_state.current_player_turn == 1
    assert new_state.turn_phase == TurnPhase.PLAY
    new_hand = new_state.players[1].hand
    new_deck = new_state.deck
    assert len(new_hand) == 2
    assert len(new_deck) == 12

    old_card, new_card = new_hand
    assert old_hand[0] == old_card
    assert old_hand[0] != new_card

    assert new_card in old_deck
    for card in new_deck:
        assert card in old_deck


def test_action_guess_2p(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=123)
    state = game.next_state(state, DRAW_ACTION)
    state = set_hand(state, 1, (Card.GUESS, Card.LOSE))
    legal_actions = game.legal_actions(state)
    assert len(legal_actions) == 8  # 7 guessable cards + 1 LOSE card action

    assert legal_actions[0] == Action(ActionType.PLAY, card=Card.GUESS, player_id=2, guess_card=Card.PEEK)
    assert legal_actions[1] == Action(ActionType.PLAY, card=Card.GUESS, player_id=2, guess_card=Card.COMPARE)
    assert legal_actions[6] == Action(ActionType.PLAY, card=Card.GUESS, player_id=2, guess_card=Card.LOSE)
    assert legal_actions[7] == Action(ActionType.PLAY, card=Card.LOSE, player_id=None, guess_card=None)

    # Ensure GUESS is not in the guessable cards
    assert not any(action.guess_card == Card.GUESS for action in legal_actions)


def test_action_guess_4p(game_4p: Infiltr8Game) -> None:
    state = game_4p.initial_state(random_seed=123)
    state = game_4p.next_state(state, DRAW_ACTION)
    state = set_hand(state, 1, (Card.GUESS, Card.LOSE))
    legal_actions = game_4p.legal_actions(state)
    assert len(legal_actions) == 22

    assert legal_actions[0] == Action(ActionType.PLAY, Card.GUESS, player_id=2, guess_card=Card.PEEK)
    assert legal_actions[-2] == Action(ActionType.PLAY, Card.GUESS, player_id=4, guess_card=Card.LOSE)
    assert legal_actions[-1] == Action(ActionType.PLAY, Card.LOSE, player_id=None, guess_card=None)


@pytest.mark.parametrize("card", [Card.PEEK, Card.COMPARE, Card.SWAP])
def test_action_target_other_player_4p(game_4p: Infiltr8Game, card: Card) -> None:
    state: Infiltr8State = game_4p.initial_state(random_seed=123)
    state = game_4p.next_state(state, DRAW_ACTION)
    state = set_hand(state, 1, (card, Card.LOSE))
    legal_actions: list[Action] = game_4p.legal_actions(state)
    assert len(legal_actions) == 4

    assert legal_actions[0] == Action(ActionType.PLAY, card=card, player_id=2, guess_card=None)
    assert legal_actions[1] == Action(ActionType.PLAY, card=card, player_id=3, guess_card=None)
    assert legal_actions[2] == Action(ActionType.PLAY, card=card, player_id=4, guess_card=None)


@pytest.mark.parametrize("card", [Card.PROTECT, Card.CONDITIONAL_DISCARD, Card.LOSE])
def test_action_no_target(game: Infiltr8Game, card: Card) -> None:
    state: Infiltr8State = game.initial_state(random_seed=123)
    state = game.next_state(state, DRAW_ACTION)
    card2 = Card.LOSE if card != Card.LOSE else Card.PROTECT  # test 2 distinct cards.
    state = set_hand(state, 1, (card, card2))
    legal_actions: list[Action] = game.legal_actions(state)
    assert len(legal_actions) == 2

    assert legal_actions[0] == Action(ActionType.PLAY, card=card, player_id=None, guess_card=None)


@pytest.mark.parametrize(
    "card2, is_force_discard",
    [
        (Card.GUESS, False),
        (Card.PEEK, False),
        (Card.COMPARE, False),
        (Card.PROTECT, False),
        (Card.SWAP, True),
        (Card.FORCE_DISCARD, True),
        (Card.LOSE, False),
    ],
)
def test_action_conditional_discard_4p(game_4p: Infiltr8Game, card2: Card, is_force_discard: bool) -> None:
    state: Infiltr8State = game_4p.initial_state(random_seed=123)
    state = game_4p.next_state(state, DRAW_ACTION)
    state = set_hand(state, 1, (Card.CONDITIONAL_DISCARD, card2))
    legal_actions: list[Action] = game_4p.legal_actions(state)

    if is_force_discard:
        assert len(legal_actions) == 1
    else:
        assert len(legal_actions) > 1
    assert Action(ActionType.PLAY, card=Card.CONDITIONAL_DISCARD, player_id=None, guess_card=None) in legal_actions


def test_next_state_draw(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=42)
    action = DRAW_ACTION
    new_state = game.next_state(state, action)

    assert len(new_state.players[1].hand) == 2
    assert len(new_state.deck) == len(state.deck) - 1
    assert new_state.turn_phase == TurnPhase.PLAY


def test_next_state_play_guess_correct(game_4p: Infiltr8Game) -> None:
    state = game_4p.initial_state(random_seed=42)
    state = set_hand(state, 1, (Card.GUESS,))
    state = set_hand(state, 2, (Card.PROTECT,))
    state = replace(state, turn_phase=TurnPhase.PLAY)

    action = Action(ActionType.PLAY, card=Card.GUESS, player_id=2, guess_card=Card.PROTECT)
    new_state = game_4p.next_state(state, action)

    assert new_state.players[2].is_out
    assert new_state.current_player_turn == 3  # Skip player 2 as they are out.
    assert new_state.turn_phase == TurnPhase.DRAW


def test_next_state_play_guess_incorrect(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=42)
    state = set_hand(state, 1, (Card.GUESS,))
    state = set_hand(state, 2, (Card.PROTECT,))
    state = replace(state, turn_phase=TurnPhase.PLAY)

    action = Action(ActionType.PLAY, card=Card.GUESS, player_id=2, guess_card=Card.PEEK)
    new_state = game.next_state(state, action)

    assert not new_state.players[2].is_out
    assert new_state.current_player_turn == 2
    assert new_state.turn_phase == TurnPhase.DRAW


def test_next_state_play_protect(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=42)
    state = set_hand(state, 1, (Card.PROTECT,))
    state = replace(state, turn_phase=TurnPhase.PLAY)

    action = Action(ActionType.PLAY, card=Card.PROTECT, player_id=None, guess_card=None)
    new_state = game.next_state(state, action)

    assert new_state.players[1].is_protected
    assert new_state.current_player_turn == 2
    assert new_state.turn_phase == TurnPhase.DRAW


def test_next_state_play_force_discard(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=42)
    state = set_hand(state, 1, (Card.FORCE_DISCARD,))
    state = set_hand(state, 2, (Card.PEEK,))
    state = replace(state, turn_phase=TurnPhase.PLAY)

    action = Action(ActionType.PLAY, card=Card.FORCE_DISCARD, player_id=2, guess_card=None)
    new_state = game.next_state(state, action)

    assert len(new_state.players[2].hand) == 1
    assert new_state.players[2].hand[0] != Card.PEEK
    assert len(new_state.discard_pile) == 2  # FORCE_DISCARD_CARD and PEEK_CARD
    assert new_state.current_player_turn == 2
    assert new_state.turn_phase == TurnPhase.DRAW


def test_next_state_play_swap(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=42)
    state = set_hand(state, 1, (Card.SWAP,))
    state = set_hand(state, 2, (Card.PEEK,))
    state = replace(state, turn_phase=TurnPhase.PLAY)

    action = Action(ActionType.PLAY, card=Card.SWAP, player_id=2, guess_card=None)
    new_state = game.next_state(state, action)

    assert new_state.players[1].hand[0] == Card.PEEK
    assert new_state.players[2].hand[0] == Card.SWAP
    assert new_state.current_player_turn == 2
    assert new_state.turn_phase == TurnPhase.DRAW


def test_serialize_action(serializer: Infiltr8Serializer) -> None:
    action = Action(ActionType.DRAW, card=None, player_id=None, guess_card=None)
    serialized = serializer.serialize_action(action)
    assert serialized == {"action_type": "DRAW", "card": None, "player_id": None, "guess_card": None}

    action = Action(ActionType.PLAY, card=Card.GUESS, player_id=2, guess_card=Card.PEEK)
    serialized = serializer.serialize_action(action)
    assert serialized == {"action_type": "PLAY", "card": "Hack", "player_id": 2, "guess_card": "Scan"}


def test_parse_action(game: Infiltr8Game, serializer: Infiltr8Serializer) -> None:
    action_data: dict[str, Any] = {"action_type": "DRAW", "card": None, "player_id": None, "guess_card": None}
    parsed = serializer.parse_action(game, action_data)
    assert parsed == Action(ActionType.DRAW, card=None, player_id=None, guess_card=None)

    action_data = {"action_type": "PLAY", "card": "Hack", "player_id": 2, "guess_card": "Scan"}
    parsed = serializer.parse_action(game, action_data)
    assert parsed == Action(ActionType.PLAY, card=Card.GUESS, player_id=2, guess_card=Card.PEEK)


def test_serialize_state(game: Infiltr8Game, serializer: Infiltr8Serializer) -> None:
    state = game.initial_state(random_seed=42)
    serialized = serializer.serialize_state(game, state)

    assert "current_player" in serialized
    assert "deck_size" in serialized
    assert "discard_pile" in serialized
    assert "players" in serialized
    assert "legal_actions" in serialized

    assert serialized["current_player"] == 1
    assert serialized["deck_size"] == 13  # 16 - 2 players - 1 removed card
    assert len(serialized["discard_pile"]) == 0
    assert len(serialized["players"]) == 2
    for player in serialized["players"].values():
        assert "is_protected" in player
        assert "is_out" in player
        assert "hand" in player
        assert isinstance(player["hand"], list)
        assert len(player["hand"]) == 1  # Initial hand size
    assert all(isinstance(action, dict) for action in serialized["legal_actions"])


def test_serialization_roundtrip(game: Infiltr8Game, serializer: Infiltr8Serializer) -> None:
    original_state = game.initial_state(random_seed=42)
    original_action = game.legal_actions(original_state)[0]

    # Serialize
    serialized_action = serializer.serialize_action(original_action)

    # Deserialize (Note: We can't fully deserialize the state, so we'll just check the action)
    deserialized_action = serializer.parse_action(game, serialized_action)

    # Check if the deserialized action matches the original
    assert deserialized_action == original_action

    # Check if all legal actions can be serialized and deserialized correctly
    for action in game.legal_actions(original_state):
        serialized = serializer.serialize_action(action)
        deserialized = serializer.parse_action(game, serialized)
        assert deserialized == action


def test_distinct_legal_actions(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=123)
    state = game.next_state(state, DRAW_ACTION)

    # Set the player's hand to have two identical GUESS cards
    state = set_hand(state, 1, (Card.GUESS, Card.GUESS))

    legal_actions = game.legal_actions(state)

    # Check that we have the correct number of distinct actions
    assert len(legal_actions) == 7  # One for each unique card that can be guessed (excluding GUESS)

    # Check that all actions are for GUESS card
    assert all(action.card == Card.GUESS for action in legal_actions)

    # Check that we have one action for each unique card that can be guessed (excluding GUESS)
    guessed_cards = set(action.guess_card for action in legal_actions)
    assert guessed_cards == set(card for card in list(Card) if card != Card.GUESS)

    # Check that all actions are unique
    assert len(set(legal_actions)) == len(legal_actions)

    # Ensure GUESS is not in the guessable cards
    assert not any(action.guess_card == Card.GUESS for action in legal_actions)


def test_legal_actions_exclude_out_players(game_4p: Infiltr8Game) -> None:
    state = game_4p.initial_state(random_seed=123)
    state = game_4p.next_state(state, DRAW_ACTION)

    # Set player 1's hand to GUESS card
    state = set_hand(state, 1, (Card.GUESS,))

    # Set player 2 as out
    updated_player2 = replace(state.players[2], is_out=True)
    updated_players = state.players.set(2, updated_player2)
    state = replace(state, players=updated_players)

    legal_actions = game_4p.legal_actions(state)

    # Check that no actions target player 2
    assert all(action.player_id != 2 for action in legal_actions if action.player_id is not None)

    # Check that we can still target players 3 and 4
    assert any(action.player_id == 3 for action in legal_actions)
    assert any(action.player_id == 4 for action in legal_actions)

    # Check that the number of actions is correct (7 guess options each for players 3 and 4)
    assert len(legal_actions) == 14


def test_legal_actions_all_other_players_out(game_4p: Infiltr8Game) -> None:
    state = game_4p.initial_state(random_seed=123)
    state = game_4p.next_state(state, DRAW_ACTION)

    # Set player 1's hand to GUESS card
    state = set_hand(state, 1, (Card.GUESS,))

    # Player 2 & 4 are out.
    for player_id in [2, 4]:
        updated_player = replace(state.players[player_id], is_out=True)
        updated_players = state.players.set(player_id, updated_player)
        state = replace(state, players=updated_players)

    legal_actions = game_4p.legal_actions(state)

    # Check that there are no legal actions for the GUESS card
    assert len(legal_actions) == 7
    assert Action(ActionType.PLAY, Card.GUESS, player_id=3, guess_card=Card.PEEK) in legal_actions
    assert Action(ActionType.PLAY, Card.GUESS, player_id=1, guess_card=Card.PEEK) not in legal_actions
    assert Action(ActionType.PLAY, Card.GUESS, player_id=2, guess_card=Card.PEEK) not in legal_actions
    assert Action(ActionType.PLAY, Card.GUESS, player_id=4, guess_card=Card.PEEK) not in legal_actions


def test_legal_actions_all_opponents_protected(game_4p: Infiltr8Game) -> None:
    state = game_4p.initial_state(random_seed=123)
    state = game_4p.next_state(state, DRAW_ACTION)

    # Set player 1's hand to two different cards
    state = set_hand(state, 1, (Card.GUESS, Card.PEEK))

    # Set all other players as protected
    for player_id in [2, 3, 4]:
        updated_player = replace(state.players[player_id], is_protected=True)
        updated_players = state.players.set(player_id, updated_player)
        state = replace(state, players=updated_players)

    legal_actions = game_4p.legal_actions(state)

    # Check that we have two DISCARD actions, one for each card
    assert len(legal_actions) == 2
    assert Action(ActionType.DISCARD, Card.GUESS, player_id=None, guess_card=None) in legal_actions
    assert Action(ActionType.DISCARD, Card.PEEK, player_id=None, guess_card=None) in legal_actions


def test_handle_discard_action(game_4p: Infiltr8Game) -> None:
    state = game_4p.initial_state(random_seed=123)
    state = game_4p.next_state(state, DRAW_ACTION)

    # Set player 1's hand to two different cards
    initial_hand = (Card.GUESS, Card.PEEK)
    state = set_hand(state, 1, initial_hand)

    # Create a discard action
    discard_action = Action(ActionType.DISCARD, Card.GUESS, player_id=None, guess_card=None)

    # Apply the discard action
    new_state = game_4p.next_state(state, discard_action)

    # Check that only one card was discarded
    assert new_state.players[1].hand == (Card.PEEK,)
    assert new_state.discard_pile[-1] == Card.GUESS

    # Check that the turn moved to the next player
    assert new_state.current_player_turn == 2
    assert new_state.turn_phase == TurnPhase.DRAW

    # Check that the action was logged
    assert new_state.action_log[-1] == f"Player 1 discarded {CARD_NAMES[Card.GUESS]}"


def test_handle_discard_action_with_duplicate_cards(game_4p: Infiltr8Game) -> None:
    state = game_4p.initial_state(random_seed=123)
    state = game_4p.next_state(state, DRAW_ACTION)

    # Set player 1's hand to two identical cards
    initial_hand = (Card.GUESS, Card.GUESS)
    state = set_hand(state, 1, initial_hand)

    # Create a discard action
    discard_action = Action(ActionType.DISCARD, Card.GUESS, player_id=None, guess_card=None)

    # Apply the discard action
    new_state = game_4p.next_state(state, discard_action)

    # Check that only one card was discarded
    assert new_state.players[1].hand == (Card.GUESS,)
    assert new_state.discard_pile[-1] == Card.GUESS

    # Check that the turn moved to the next player
    assert new_state.current_player_turn == 2
    assert new_state.turn_phase == TurnPhase.DRAW

    # Check that the action was logged
    assert new_state.action_log[-1] == f"Player 1 discarded {CARD_NAMES[Card.GUESS]}"
