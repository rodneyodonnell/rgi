from dataclasses import dataclass
from typing import Any, Literal
from typing_extensions import override

import torch

from rgi.core import base
from rgi.core.base import Game, GameSerializer


TPlayerId = Literal[1, 2]
TPosition = tuple[int, int]  # TODOO: remove.


@dataclass(frozen=True)
class OthelloState:
    board: torch.Tensor  # 8x8 tensor, 0 for empty, 1 for black, 2 for white
    current_player: TPlayerId
    is_terminal: bool


@dataclass
class BatchOthelloState(base.Batch[OthelloState]):
    board: torch.Tensor
    current_player: torch.Tensor


Action = int
BatchAction = base.PrimitiveBatch[Action]


class OthelloGame(Game[OthelloState, TPlayerId, Action]):
    """Othello game implementation.

    Actions are positions on the board where the player can place a disc.
    """

    def __init__(self, board_size: int = 8):
        self.board_size = board_size
        self._all_positions = [(row, col) for row in range(self.board_size) for col in range(self.board_size)]
        self._directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    @override
    def initial_state(self) -> OthelloState:
        board = torch.zeros((self.board_size, self.board_size), dtype=torch.int8)
        mid = self.board_size // 2
        board[mid - 1, mid - 1] = 2
        board[mid - 1, mid] = 1
        board[mid, mid - 1] = 1
        board[mid, mid] = 2
        return OthelloState(board=board, current_player=1, is_terminal=False)

    @override
    def current_player_id(self, state: OthelloState) -> TPlayerId:
        return state.current_player

    def next_player(self, player: TPlayerId) -> TPlayerId:
        return 1 if player == 2 else 2

    @override
    def all_player_ids(self, game_state: OthelloState) -> list[TPlayerId]:
        return [1, 2]

    @override
    def legal_actions(self, game_state: OthelloState) -> list[Action]:
        return self._get_legal_moves(game_state, game_state.current_player)

    @override
    def all_actions(self) -> list[Action]:
        return self._all_positions

    def _get_legal_moves(self, game_state: OthelloState, player: TPlayerId) -> list[Action]:
        opponent = self.next_player(player)
        legal_moves = []

        for row in range(self.board_size):
            for col in range(self.board_size):
                if state.board[row, col] != 0:
                    continue  # Skip occupied positions
                if self._would_flip(state, (row, col), player, opponent):
                    legal_moves.append((row, col))
        return legal_moves

    def _would_flip(self, state: OthelloState, position: TPosition, player: TPlayerId, opponent: TPlayerId) -> bool:
        for dr, dc in self.directions:
            r, c = position[0] + dr, position[1] + dc
            found_opponent = False
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if state.board[r, c] == 0:
                    break
                elif state.board[r, c] == opponent:
                    found_opponent = True
                elif state.board[r, c] == player:
                    if found_opponent:
                        return True
                    else:
                        break
                r += dr
                c += dc
        return False

    @override
    def next_state(self, game_state: OthelloState, action: Action) -> OthelloState:
        if action not in self.legal_actions(game_state):
            raise ValueError(f"Invalid move: {action} is not a legal action.")

        player = game_state.current_player
        opponent = self.next_player(player)
        new_board = game_state.board.clone()

        positions_to_flip = self._get_positions_to_flip(game_state, action, player, opponent)

        # Place the new disc
        new_board[action[0], action[1]] = player

        # Flip the opponent's discs
        for pos in positions_to_flip:
            new_board[pos[0], pos[1]] = player

        # Determine next player
        next_player = self.next_player(player)

        # If next player has no legal moves, current player plays again
        next_state = OthelloState(new_board, next_player, is_terminal=False)
        if not self._get_legal_moves(next_state, next_player):
            if self._get_legal_moves(next_state, player):
                next_player = player
            else:
                # No moves for either player; the game ends
                return OthelloState(new_board, player, is_terminal=True)

        return OthelloState(new_board, next_player, is_terminal=False)

    def _get_positions_to_flip(
        self, state: OthelloState, position: TPosition, player: TPlayerId, opponent: TPlayerId
    ) -> list[TPosition]:
        positions_to_flip = []

        for dr, dc in self.directions:
            r, c = position[0] + dr, position[1] + dc
            temp_positions = []

            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if state.board[r, c] == 0:
                    break
                elif state.board[r, c] == opponent:
                    temp_positions.append((r, c))
                elif state.board[r, c] == player:
                    if temp_positions:
                        positions_to_flip.extend(temp_positions)
                    break
                r += dr
                c += dc

        return positions_to_flip

    @override
    def is_terminal(self, game_state: OthelloState) -> bool:
        return game_state.is_terminal

    @override
    def reward(self, game_state: OthelloState, player_id: TPlayerId) -> float:
        if not self.is_terminal(game_state):
            return 0.0
        player_count = torch.sum(game_state.board == player_id).item()
        opponent_count = torch.sum(game_state.board == self.next_player(player_id)).item()
        if player_count > opponent_count:
            return 1.0
        elif player_count < opponent_count:
            return -1.0
        else:
            return 0.0  # Draw

    @override
    def pretty_str(self, game_state: OthelloState) -> str:
        def cell_to_str(cell: int) -> str:
            return "●" if cell == 1 else "○" if cell == 2 else "."

        rows = []
        for row in range(self.board_size):
            row_str = " ".join(cell_to_str(game_state.board[row, col].item()) for col in range(self.board_size))
            rows.append(row_str)

        return "\n".join(rows)

    def parse_board(self, board_str: str, current_player: TPlayerId, is_terminal: bool) -> OthelloState:
        """Parses a board string into an OthelloState."""
        board = torch.zeros((self.board_size, self.board_size), dtype=torch.int8)
        rows = board_str.strip().split("\n")
        for r, row in enumerate(rows):
            cells = row.strip().split()
            for c, cell in enumerate(cells):
                if cell == "●":
                    board[r, c] = 1
                elif cell == "○":
                    board[r, c] = 2
        return OthelloState(board=board, current_player=current_player, is_terminal=is_terminal)


class OthelloSerializer(GameSerializer[OthelloGame, OthelloState, Action]):
    @override
    def serialize_state(self, game: OthelloGame, state: OthelloState) -> dict[str, Any]:
        return {
            "rows": game.board_size,
            "columns": game.board_size,
            "state": state.board.tolist(),
            "current_player": state.current_player,
            "legal_actions": game.legal_actions(state),
            "is_terminal": game.is_terminal(state),
        }

    @override
    def parse_action(self, game: OthelloGame, action_data: dict[str, Any]) -> Action:
        row = action_data.get("row")
        col = action_data.get("col")
        if row is None or col is None:
            raise ValueError("Action data must include 'row' and 'col'")
        return (int(row), int(col))
