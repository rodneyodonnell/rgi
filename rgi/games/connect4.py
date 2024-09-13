from typing import Literal, Optional
from typing_extensions import override
from immutables import Map
from rgi.core.base import Game, TPlayerId, TAction


class Connect4State:
    def __init__(
        self, board: Map[tuple[int, int], int], current_player: Literal[1, 2], winner: Optional[Literal[1, 2]] = None
    ):
        # Immutable map of board. Indexed by (row,column). board[(1,1)] is bottom left corner.
        self.board = board
        self.current_player = current_player
        self.winner = winner

    def __repr__(self) -> str:
        return f"Connect4State(board={self.board}, current_player={self.current_player}, winner={self.winner})"


class Connect4Game(Game[Connect4State, TPlayerId, TAction]):
    """Connect 4 game implementation.

    Actions are column numbers (1-7) where the player can drop a piece.
    """

    def __init__(self, width: int = 7, height: int = 6, connect: int = 4):
        self.width = width
        self.height = height
        self.connect = connect
        self._all_column_ids = list(range(1, width + 1))
        self._all_row_ids = list(range(1, height + 1))

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
        return [col for col in self._all_column_ids if (self.height, col) not in state.board]

    @override
    def next_state(self, state: Connect4State, action: int) -> Connect4State:
        """Find the lowest empty row in the selected column and return the updated game state."""
        if action not in self.legal_actions(state):
            raise ValueError(f"Invalid move: Invalid column '{action}' no in {self._all_column_ids}")

        for row in range(1, self.height + 1):
            if (row, action) not in state.board:
                new_board = state.board.set((row, action), state.current_player)
                winner = self._calculate_winner(new_board, action, row, state.current_player)
                next_player = 2 if state.current_player == 1 else 1
                return Connect4State(board=new_board, current_player=next_player, winner=winner)

        raise ValueError("Invalid move: column is full")

    def _calculate_winner(self, board: Map[tuple[int, int], int], col: int, row: int, player: int) -> Optional[int]:
        """Check if the last move made at (row, col) by 'player' wins the game."""
        # fmt: off
        directions = [
            (1, 0), (-1, 0),  # Vertical
            (0, 1), (0, -1),  # Horizontal
            (1, 1), (-1, -1),  # Diagonal /
            (1, -1), (-1, 1),  # Diagonal \
        ]
        # fmt: on

        def count_in_direction(delta_row: int, delta_col: int) -> int:
            """Count consecutive pieces in one direction."""
            count = 0
            current_row, current_col = row + delta_row, col + delta_col
            while 1 <= current_row <= self.height and 1 <= current_col <= self.width:
                if board.get((current_row, current_col)) == player:
                    count += 1
                    current_row += delta_row
                    current_col += delta_col
                else:
                    break
            return count

        # Check all directions from the last move
        for delta_row, delta_col in directions:
            # +1 to include the current square.
            consecutive_count = count_in_direction(delta_row, delta_col) + 1
            if consecutive_count >= self.connect:
                return player

        # Check all directions from the last move
        # for direction in directions:
        #     consecutive_count = 1  # Include the last move itself
        #     for delta_col, delta_row in direction:
        #         consecutive_count = count_in_direction(delta_col, delta_row)
        #     if consecutive_count >= self.connect:
        #         return player  # Current player wins

        return None  # No winner yet

    @override
    def is_terminal(self, state: Connect4State) -> bool:
        if state.winner is not None:
            return True
        return all((self.height, col) in state.board for col in self._all_column_ids)

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
                "|" + "|".join(" ●○"[state.board.get((row, col), 0)] for col in self._all_column_ids) + "|"
                for row in reversed(self._all_row_ids)  # Start from the top row and work down
            )
            + "\n+"
            + "-+" * self.width
        )
