from rgi.core.base import Player, Game, TGameState, TAction
from typing import Callable, Generic, TypeVar
from typing_extensions import override

TGame = TypeVar('TGame', bound=Game)

class HumanPlayer(Player[TGameState, None, TAction], Generic[TGame, TGameState, TAction]):
    def __init__(self, game: TGame, action_input_func: Callable[[list[TAction]], TAction]):
        self.game = game
        self.action_input_func = action_input_func

    @override
    def select_action(self, game_state: TGameState, legal_actions: list[TAction]) -> TAction:
        print("Current game state:")
        print(self.game.pretty_str(game_state))
        print("Legal actions:")
        for i, action in enumerate(legal_actions):
            print(f"{i}: {action}")
        
        return self.action_input_func(legal_actions)
    @override
    def update_state(self, game_state: TGameState, action: TAction):
        # Human player doesn't need to maintain any state
        pass