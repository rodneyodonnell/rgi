from rgi.core.player import Player
from rgi.core.game import Game, TState, TAction, TPlayer

class HumanPlayer(Player[TState, TAction, None]):
    def select_action(self, game: Game[TState, TAction, TPlayer], state: TState) -> TAction:
        print("\nCurrent game state:")
        print(game.state_to_string(state))
        
        legal_actions = game.legal_actions(state)
        print("\nLegal actions:")
        for i, action in enumerate(legal_actions):
            print(f"{i}: {game.action_to_string(action)}")
        
        while True:
            try:
                choice = int(input("\nEnter the number of your chosen action: "))
                if 0 <= choice < len(legal_actions):
                    return legal_actions[choice]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def update(self, state: TState, action: TAction, new_state: TState) -> None:
        pass  # HumanPlayer doesn't need to update internal state

    def get_state(self) -> None:
        return None  # HumanPlayer has no internal state

    def set_state(self, state: None) -> None:
        pass  # HumanPlayer has no internal state to set