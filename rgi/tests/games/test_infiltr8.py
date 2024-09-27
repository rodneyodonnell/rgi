# rgi/tests/games/test_infiltr8.py

from dataclasses import replace
import pytest
from rgi.games.infiltr8 import Infiltr8Game, Infiltr8State, Card, Action, ActionType, TurnPhase
from rgi.games import infiltr8

# pylint: disable=redefined-outer-name  # pytest fixtures trigger this false positive


@pytest.fixture
def game() -> Infiltr8Game:
    return Infiltr8Game(num_players=2)


@pytest.fixture
def game_4p() -> Infiltr8Game:
    return Infiltr8Game(num_players=4)


DRAW_ACTION = Action(ActionType.DRAW, card=None, player_id=None, value=None)


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
    state = set_hand(state, 1, (infiltr8.GUESS_CARD, infiltr8.LOSE_CARD))
    legal_actions = game.legal_actions(state)
    assert len(legal_actions) == 9

    assert legal_actions[0] == Action(ActionType.PLAY, card=infiltr8.GUESS_CARD, player_id=2, value=1)
    assert legal_actions[1] == Action(ActionType.PLAY, card=infiltr8.GUESS_CARD, player_id=2, value=2)
    assert legal_actions[7] == Action(ActionType.PLAY, card=infiltr8.GUESS_CARD, player_id=2, value=8)
    assert legal_actions[8] == Action(ActionType.PLAY, card=infiltr8.LOSE_CARD, player_id=None, value=None)


def test_action_guess_4p(game_4p: Infiltr8Game) -> None:
    state = game_4p.initial_state(random_seed=123)
    state = game_4p.next_state(state, DRAW_ACTION)
    state = set_hand(state, 1, (infiltr8.GUESS_CARD, infiltr8.LOSE_CARD))
    legal_actions = game_4p.legal_actions(state)
    assert len(legal_actions) == 25

    assert legal_actions[0] == Action(ActionType.PLAY, card=infiltr8.GUESS_CARD, player_id=2, value=1)
    assert legal_actions[23] == Action(ActionType.PLAY, card=infiltr8.GUESS_CARD, player_id=4, value=8)
    assert legal_actions[24] == Action(ActionType.PLAY, card=infiltr8.LOSE_CARD, player_id=None, value=None)


@pytest.mark.parametrize("card", [infiltr8.PEEK_CARD, infiltr8.COMPARE_CARD, infiltr8.SWAP_CARD])
def test_action_target_other_player_4p(game_4p: Infiltr8Game, card: Card) -> None:  # Added card parameter type
    state: Infiltr8State = game_4p.initial_state(random_seed=123)
    state = game_4p.next_state(state, DRAW_ACTION)
    state = set_hand(state, 1, (card, infiltr8.LOSE_CARD))
    legal_actions: list[Action] = game_4p.legal_actions(state)
    assert len(legal_actions) == 4

    assert legal_actions[0] == Action(ActionType.PLAY, card=card, player_id=2, value=None)
    assert legal_actions[1] == Action(ActionType.PLAY, card=card, player_id=3, value=None)
    assert legal_actions[2] == Action(ActionType.PLAY, card=card, player_id=4, value=None)


@pytest.mark.parametrize("card", [infiltr8.PROTECT_CARD, infiltr8.CONDITIONAL_DISCARD_CARD, infiltr8.LOSE_CARD])
def test_action_no_target(game: Infiltr8Game, card: Card) -> None:
    state: Infiltr8State = game.initial_state(random_seed=123)
    state = game.next_state(state, DRAW_ACTION)
    state = set_hand(state, 1, (card, card))
    legal_actions: list[Action] = game.legal_actions(state)
    assert len(legal_actions) == 2

    assert legal_actions[0] == Action(ActionType.PLAY, card=card, player_id=None, value=None)


@pytest.mark.parametrize(
    "card2, is_force_discard",
    [
        (infiltr8.GUESS_CARD, False),
        (infiltr8.PEEK_CARD, False),
        (infiltr8.COMPARE_CARD, False),
        (infiltr8.PROTECT_CARD, False),
        (infiltr8.SWAP_CARD, True),
        (infiltr8.FORCE_DISCARD_CARD, True),
        (infiltr8.LOSE_CARD, False),
    ],
)
def test_action_conditional_discard_4p(game_4p: Infiltr8Game, card2: Card, is_force_discard: bool) -> None:
    state: Infiltr8State = game_4p.initial_state(random_seed=123)
    state = game_4p.next_state(state, DRAW_ACTION)
    state = set_hand(state, 1, (infiltr8.CONDITIONAL_DISCARD_CARD, card2))
    legal_actions: list[Action] = game_4p.legal_actions(state)

    if is_force_discard:
        assert len(legal_actions) == 1
    else:
        assert len(legal_actions) > 1
    assert Action(ActionType.PLAY, card=infiltr8.CONDITIONAL_DISCARD_CARD, player_id=None, value=None) in legal_actions


def test_next_state_draw(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=42)
    action = DRAW_ACTION
    new_state = game.next_state(state, action)

    assert len(new_state.players[1].hand) == 2
    assert len(new_state.deck) == len(state.deck) - 1
    assert new_state.turn_phase == TurnPhase.PLAY


def test_next_state_play_guess_correct(game_4p: Infiltr8Game) -> None:
    state = game_4p.initial_state(random_seed=42)
    state = set_hand(state, 1, (infiltr8.GUESS_CARD,))
    state = set_hand(state, 2, (infiltr8.PROTECT_CARD,))
    state = replace(state, turn_phase=TurnPhase.PLAY)

    action = Action(ActionType.PLAY, card=infiltr8.GUESS_CARD, player_id=2, value=infiltr8.PROTECT_CARD.effect.value)
    new_state = game_4p.next_state(state, action)

    assert new_state.players[2].is_out
    assert new_state.current_player_turn == 3  # Skip player 2 as they are out.
    assert new_state.turn_phase == TurnPhase.DRAW


def test_next_state_play_guess_incorrect(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=42)
    state = set_hand(state, 1, (infiltr8.GUESS_CARD,))
    state = set_hand(state, 2, (infiltr8.PROTECT_CARD,))
    state = replace(state, turn_phase=TurnPhase.PLAY)

    action = Action(ActionType.PLAY, card=infiltr8.GUESS_CARD, player_id=2, value=infiltr8.PEEK_CARD.effect.value)
    new_state = game.next_state(state, action)

    assert not new_state.players[2].is_out
    assert new_state.current_player_turn == 2
    assert new_state.turn_phase == TurnPhase.DRAW


def test_next_state_play_protect(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=42)
    state = set_hand(state, 1, (infiltr8.PROTECT_CARD,))
    state = replace(state, turn_phase=TurnPhase.PLAY)

    action = Action(ActionType.PLAY, card=infiltr8.PROTECT_CARD, player_id=None, value=None)
    new_state = game.next_state(state, action)

    assert new_state.players[1].is_protected
    assert new_state.current_player_turn == 2
    assert new_state.turn_phase == TurnPhase.DRAW


def test_next_state_play_force_discard(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=42)
    state = set_hand(state, 1, (infiltr8.FORCE_DISCARD_CARD,))
    state = set_hand(state, 2, (infiltr8.PEEK_CARD,))
    state = replace(state, turn_phase=TurnPhase.PLAY)

    action = Action(ActionType.PLAY, card=infiltr8.FORCE_DISCARD_CARD, player_id=2, value=None)
    new_state = game.next_state(state, action)

    assert len(new_state.players[2].hand) == 1
    assert new_state.players[2].hand[0] != infiltr8.PEEK_CARD
    assert len(new_state.discard_pile) == 2  # FORCE_DISCARD_CARD and PEEK_CARD
    assert new_state.current_player_turn == 2
    assert new_state.turn_phase == TurnPhase.DRAW


def test_next_state_play_swap(game: Infiltr8Game) -> None:
    state = game.initial_state(random_seed=42)
    state = set_hand(state, 1, (infiltr8.SWAP_CARD,))
    state = set_hand(state, 2, (infiltr8.PEEK_CARD,))
    state = replace(state, turn_phase=TurnPhase.PLAY)

    action = Action(ActionType.PLAY, card=infiltr8.SWAP_CARD, player_id=2, value=None)
    new_state = game.next_state(state, action)

    assert new_state.players[1].hand[0] == infiltr8.PEEK_CARD
    assert new_state.players[2].hand[0] == infiltr8.SWAP_CARD
    assert new_state.current_player_turn == 2
    assert new_state.turn_phase == TurnPhase.DRAW
