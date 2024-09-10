from typing import Literal
from typing_extensions import override
from immutables import Map
from rgi.core.base import Game, TPlayerId, TAction

class Connect4State:
    def __init__(self, board: Map[tuple[int, int], int], current_player: Literal[1, 2]):
        self.board = board
        self.current_player = current_player

    def __repr__(self):
        return f"Connect4State(board={self.board}, current_player={self.current_player})"

class Connect4Game(Game[Connect4State, Literal[1, 2], int]):
    def __init__(self, width: int = 7, height: int = 6, connect: int = 4):
        self.width = width
        self.height = height
        self.connect = connect

    @override
    def initial_state(self) -> Connect4State:
        return Connect4State(Map({(row, col): 0 for row in range(self.height) for col in range(self.width)}), current_player=1)

    @override
    def current_player_id(self, state: Connect4State) -> Literal[1, 2]:
        return state.current_player

    @override
    def all_player_ids(self, state: Connect4State) -> list[Literal[1, 2]]:
        return [1, 2]

    @override
    def legal_actions(self, state: Connect4State) -> list[int]:
        return [col for col in range(self.width) if state.board.get((0, col)) == 0]

    @override
    def next_state(self, state: Connect4State, action: int) -> Connect4State:
        if action not in self.legal_actions(state):
            raise ValueError("Illegal action")

        new_board = state.board
        for row in range(self.height - 1, -1, -1):
            if new_board.get((row, action)) == 0:
                new_board = new_board.set((row, action), state.current_player)
                break

        return Connect4State(
            board=new_board,
            current_player=3 - state.current_player  # Switch player (1 -> 2, 2 -> 1)
        )

    @override
    def is_terminal(self, state: Connect4State) -> bool:
        return self._check_winner(state) is not None or all(state.board.get((0, col)) != 0 for col in range(self.width))

    @override
    def reward(self, state: Connect4State, player_id: Literal[1, 2]) -> float:
        winner = self._check_winner(state)
        if winner is None:
            return 0
        return 1 if winner == player_id else -1

    def _check_winner(self, state: Connect4State) -> Literal[1, 2] | None:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonal down, Diagonal up
        for row in range(self.height):
            for col in range(self.width):
                if state.board.get((row, col)) == 0:
                    continue
                for dx, dy in directions:
                    if all(0 <= row + i*dy < self.height and 0 <= col + i*dx < self.width and 
                           state.board.get((row, col)) == state.board.get((row + i*dy, col + i*dx)) 
                           for i in range(self.connect)):
                        return state.board.get((row, col))
        return None

    def __str__(self) -> str:
        return f"Connect4Game(width={self.width}, height={self.height}, connect={self.connect})"

    @override
    def pretty_str(self, state: Connect4State) -> str:
        return "\n".join(
            "|" + "|".join(" ●○"[state.board.get((row, col), 0)] for col in range(self.width)) + "|"
            for row in range(self.height)
        ) + "\n+" + "-+" * self.width