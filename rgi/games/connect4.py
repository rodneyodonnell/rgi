from typing import Literal, Optional
from typing_extensions import override
from immutables import Map
from rgi.core.base import Game, TPlayerId, TAction


class Connect4State:
    def __init__(
        self, board: Map[tuple[int, int], int], current_player: Literal[1, 2], winner: Optional[Literal[1, 2]] = None
    ):
        self.board = board
        self.current_player = current_player
        self.winner = winner

    def __repr__(self) -> str:
        return f"Connect4State(board={self.board}, current_player={self.current_player}, winner={self.winner})"


class Connect4Game(Game[Connect4State, TPlayerId, TAction]):
    def __init__(self, width: int = 7, height: int = 6, connect: int = 4):
        self.width = width
        self.height = height
        self.connect = connect

    @override
    def initial_state(self) -> Connect4State:
        return Connect4State(board=Map(), current_player=1)

    @override
    def current_player_id(self, state: Connect4State) -> TPlayerId:
        return state.current_player

    @override
    def all_player_ids(self, state: Connect4State) -> list[TPlayerId]:
        """Return a list of all player IDs."""
        return [1, 2]

    @override
    def legal_actions(self, state: Connect4State) -> list[int]:
        return [col for col in range(self.width) if (col, self.height - 1) not in state.board]

    @override
    def next_state(self, state: Connect4State, action: int) -> Connect4State:
        """Find the lowest empty row in the selected column and return the updated game state."""
        for row in range(self.height):
            if (action, row) not in state.board:
                new_board = state.board.set((action, row), state.current_player)
                winner = self._calculate_winner(new_board, action, row, state.current_player)
                next_player = 2 if state.current_player == 1 else 1
                return Connect4State(board=new_board, current_player=next_player, winner=winner)

        raise ValueError("Invalid move: column is full")

    def _calculate_winner(self, board: Map[tuple[int, int], int], col: int, row: int, player: int) -> Optional[int]:
        """Check if the last move made at (col, row) by 'player' wins the game."""
        directions = [
            [(1, 0), (-1, 0)],  # Horizontal
            [(0, 1), (0, -1)],  # Vertical
            [(1, 1), (-1, -1)],  # Diagonal /
            [(1, -1), (-1, 1)],  # Diagonal \
        ]

        def count_in_direction(delta_col: int, delta_row: int) -> int:
            """Count consecutive pieces in one direction."""
            count = 0
            current_col, current_row = col + delta_col, row + delta_row
            while 0 <= current_col < self.width and 0 <= current_row < self.height:
                if board.get((current_col, current_row)) == player:
                    count += 1
                    current_col += delta_col
                    current_row += delta_row
                else:
                    break
            return count

        # Check all directions from the last move
        for direction in directions:
            consecutive_count = 1  # Include the last move itself
            for delta_col, delta_row in direction:
                consecutive_count += count_in_direction(delta_col, delta_row)
            if consecutive_count >= self.connect:
                return player  # Current player wins

        return None  # No winner yet

    @override
    def is_terminal(self, state: Connect4State) -> bool:
        if state.winner is not None:
            return True
        return all((col, self.height - 1) in state.board for col in range(self.width))

    @override
    def reward(self, state: Connect4State, player_id: int) -> float:
        if state.winner == player_id:
            return 1.0
        elif state.winner is not None:
            return -1.0
        return 0.0

    @override
    def pretty_str(self, state: Connect4State) -> str:
        return (
            "\n".join(
                "|" + "|".join(" ●○"[state.board.get((col, row), 0)] for col in range(self.width)) + "|"
                for row in reversed(range(self.height))  # Start from the top row and work down
            )
            + "\n+"
            + "-+" * self.width
        )
