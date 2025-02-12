from typing import Literal, Sequence

from typing_extensions import override

from rgi.core.base import ActionResult, Player, TAction, TGame, TGameState

_INDEX_PREFIX = "i:"

TPlayerState = Literal[None]
TPlayerData = Literal[None]


class HumanPlayer(Player[TGameState, TPlayerState, TAction, TPlayerData]):
    def __init__(self, game: TGame):
        self.game = game

    @override
    def select_action(
        self, game_state: TGameState, legal_actions: Sequence[TAction]
    ) -> ActionResult[TAction, TPlayerData]:
        action = self._select_action(game_state, legal_actions)
        return ActionResult(action, None)

    def _select_action(self, game_state: TGameState, legal_actions: Sequence[TAction]) -> TAction:
        while True:
            print("Current game state:")
            print(self.game.pretty_str(game_state))
            print("Legal actions:")
            for i, action in enumerate(legal_actions):
                print(f"{_INDEX_PREFIX}{i+1} or {action}")

            choice_str = input("Enter the index of your chosen action: ").strip()
            # i:x is verbose & safer way of choosing an action.
            if choice_str.startswith(_INDEX_PREFIX):
                return legal_actions[int(choice_str[len(_INDEX_PREFIX) :]) - 1]
            # User can type the action directly
            for i, action in enumerate(legal_actions):
                if str(action) == choice_str:
                    return action
            # User can choose the action number without the i: prefix
            try:
                return legal_actions[int(choice_str) - 1]
            except (ValueError, IndexError):
                pass
            print("##\n##\n## Invalid input. Please enter a valid action action.\n##\n##")
