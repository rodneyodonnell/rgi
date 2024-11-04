from dataclasses import dataclass

from typing import Sequence, Any, Optional
from typing_extensions import override

import numpy as np
from numpy.typing import NDArray

from rgi.core import base


@dataclass
class Connect4State:
    board: NDArray[np.int8]  # (height, width)
    current_player: int
    winner: Optional[int] = None  # The winner, if the game has ended


GameState = Connect4State
Action = int
PlayerId = int


class Connect4Game(base.Game[GameState, Action]):
    """Connect 4 game implementation.

    Actions are column numbers (1-7) where the player can drop a piece.
    """

    def __init__(self, width: int = 7, height: int = 6, connect_length: int = 4):
        self.width = width
        self.height = height
        self.connect_length = connect_length
        self._all_column_ids = tuple(range(1, width + 1))
        self._all_row_ids = tuple(range(1, height + 1))

    @override
    def initial_state(self) -> GameState:
        return GameState(board=np.zeros([self.height, self.width], dtype=np.int8), current_player=1, winner=None)

    @override
    def current_player_id(self, game_state: GameState) -> int:
        return game_state.current_player

    @override
    def num_players(self, game_state: GameState) -> int:
        return 2

    @override
    def legal_actions(self, game_state: GameState) -> Sequence[Action]:
        return tuple(col + 1 for col in range(self.width) if game_state.board[0, col] == 0)

    @override
    def all_actions(self) -> Sequence[Action]:
        return self._all_column_ids

    @override
    def next_state(self, game_state: GameState, action: Action) -> GameState:
        """Find the lowest empty row in the selected column and return the updated game state."""
        if action not in self.legal_actions(game_state):
            raise ValueError(f"Invalid move: Invalid column '{action}' not in {self._all_column_ids}")

        column = action - 1  # Convert 1-based action to 0-based column index
        row = np.nonzero(game_state.board[:, column] == 0)[0][-1]

        new_board = game_state.board.copy()
        new_board[row, column] = game_state.current_player

        winner = game_state.winner or self._calculate_winner(new_board, column, row, game_state.current_player)
        next_player: PlayerId = 2 if game_state.current_player == 1 else 1

        return GameState(board=new_board, current_player=next_player, winner=winner)

    def _calculate_winner(self, board: NDArray[np.int8], col: int, row: int, player: PlayerId) -> Optional[PlayerId]:
        """Check if the last move made at (row, col) by 'player' wins the game."""
        directions = [
            (0, 1),  # Horizontal
            (1, 0),  # Vertical
            (1, 1),  # Diagonal /
            (1, -1),  # Diagonal \
        ]

        for dr, dc in directions:
            count = 1
            for factor in [-1, 1]:
                r, c = row + dr * factor, col + dc * factor
                while 0 <= r < self.height and 0 <= c < self.width and board[r, c] == player:
                    count += 1
                    r, c = r + dr * factor, c + dc * factor
                    if count >= self.connect_length:
                        return player

        return None  # No winner yet

    @override
    def is_terminal(self, game_state: GameState) -> bool:
        if game_state.winner is not None:
            return True
        return np.all(game_state.board != 0).item()

    @override
    def reward(self, game_state: GameState, player_id: PlayerId) -> float:
        if game_state.winner == player_id:
            return 1.0
        elif game_state.winner is not None:
            return -1.0
        return 0.0

    @override
    def pretty_str(self, game_state: GameState) -> str:
        symbols = [" ", "●", "○"]
        return (
            "\n".join("|" + "|".join(symbols[int(cell)] for cell in row) + "|" for row in game_state.board)
            + "\n+"
            + "-+" * self.width
        )

    def parse_board(self, board_str: str, current_player: PlayerId) -> GameState:
        """Parses the output of pretty_str into a GameState."""
        rows = board_str.strip().split("\n")[:-1]  # Skip the bottom border row
        board = np.zeros((self.height, self.width), dtype=np.int8)
        for r, row in enumerate(rows):
            row_cells = row.strip().split("|")[1:-1]  # Extract cells between borders
            for c, cell in enumerate(row_cells):
                if cell == "●":
                    board[r, c] = 1  # Player 1
                elif cell == "○":
                    board[r, c] = 2  # Player 2
        return GameState(board=board, current_player=current_player)


class Connect4Serializer(base.GameSerializer[Connect4Game, GameState, Action]):
    @override
    def serialize_state(self, game: Connect4Game, game_state: GameState) -> dict[str, Any]:
        """Serialize the game state to a dictionary for frontend consumption."""
        board = game_state.board.tolist()
        return {
            "rows": game.height,
            "columns": game.width,
            "state": board,
            "current_player": int(game_state.current_player),
            "is_terminal": game.is_terminal(game_state),
        }

    @override
    def parse_action(self, game: Connect4Game, action_data: dict[str, Any]) -> Action:
        """Parse an action from frontend data."""
        column = action_data.get("column")
        if column is None:
            raise ValueError("Action data must include 'column'")
        if not isinstance(column, int):
            raise ValueError("Column must be an integer")
        return Action(column)
