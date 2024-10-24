from dataclasses import dataclass
from typing import Any, Literal
from typing_extensions import override

from immutables import Map
import torch
from rgi.core.base import Game, GameSerializer


TPlayerId = Literal[1, 2]
TAction = tuple[int, int]
TPosition = tuple[int, int]


@dataclass(frozen=True)
class OthelloState:
    board: Map[TPosition, int]  # Indexed by (row,column). board[(1,1)] is bottom left corner.
    current_player: TPlayerId  # The current player
    is_terminal: bool  # The winner, if the game has ended


class OthelloGame(Game[OthelloState, TPlayerId, TAction]):
    """Othello game implementation.

    Actions are positions on the board where the player can place a disc.
    """

    def __init__(self, board_size: int = 8):
        self.board_size = board_size
        self._all_positions = [(row, col) for row in range(1, board_size + 1) for col in range(1, board_size + 1)]
        # Directions are (delta_row, delta_col)
        self.directions = [
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
        board: Map[TPosition, int] = Map()
        mid = self.board_size // 2
        # Set up the initial four discs
        board = board.set((mid, mid), 2)  # White
        board = board.set((mid + 1, mid + 1), 2)  # White
        board = board.set((mid, mid + 1), 1)  # Black
        board = board.set((mid + 1, mid), 1)  # Black
        return OthelloState(board=board, current_player=1, is_terminal=False)  # Black starts

    @override
    def current_player_id(self, state: OthelloState) -> TPlayerId:
        return state.current_player

    def next_player(self, player: TPlayerId) -> TPlayerId:
        return 1 if player == 2 else 2

    @override
    def all_player_ids(self, state: OthelloState) -> list[TPlayerId]:
        return [1, 2]

    @override
    def legal_actions(self, state: OthelloState) -> list[TAction]:
        return self._get_legal_moves(state, state.current_player)

    @override
    def all_actions(self) -> list[TAction]:
        return self._all_positions

    def _get_legal_moves(self, state: OthelloState, player: TPlayerId) -> list[TAction]:
        opponent = self.next_player(player)
        legal_moves = []

        for position in self._all_positions:
            if position in state.board:
                continue  # Skip occupied positions
            if self._would_flip(state, position, player, opponent):
                legal_moves.append(position)
        return legal_moves

    def _would_flip(
        self,
        state: OthelloState,
        position: TPosition,
        player: TPlayerId,
        opponent: TPlayerId,
    ) -> bool:
        for delta_row, delta_col in self.directions:
            row, col = position
            row += delta_row
            col += delta_col
            found_opponent = False
            while 1 <= row <= self.board_size and 1 <= col <= self.board_size:
                if (row, col) not in state.board:
                    break
                elif state.board[(row, col)] == opponent:
                    found_opponent = True
                elif state.board[(row, col)] == player:
                    if found_opponent:
                        return True
                    else:
                        break
                else:
                    break
                row += delta_row
                col += delta_col
        return False

    @override
    def next_state(self, state: OthelloState, action: TAction) -> OthelloState:
        _legal_actions = self.legal_actions(state)
        if action not in _legal_actions:
            raise ValueError(f"Invalid move: {action} is not a legal action {_legal_actions}.")

        player = state.current_player
        opponent = self.next_player(player)
        new_board = state.board

        positions_to_flip = self._get_positions_to_flip(state, action, player, opponent)

        # Place the new disc
        new_board = new_board.set(action, player)

        # Flip the opponent's discs
        for pos in positions_to_flip:
            new_board = new_board.set(pos, player)

        # Determine next player
        next_player = self.next_player(player)

        is_terminal = False
        # If next player has no legal moves, current player plays again
        if not self._get_legal_moves(OthelloState(new_board, next_player, is_terminal=False), next_player):
            if self._get_legal_moves(OthelloState(new_board, player, is_terminal=False), player):
                next_player = player
            else:
                # No moves for either player; the game will end
                is_terminal = True

        return OthelloState(board=new_board, current_player=next_player, is_terminal=is_terminal)

    def _get_positions_to_flip(
        self,
        state: OthelloState,
        position: TPosition,
        player: TPlayerId,
        opponent: TPlayerId,
    ) -> list[TPosition]:
        positions_to_flip = []

        for delta_row, delta_col in self.directions:
            row, col = position
            row += delta_row
            col += delta_col
            temp_positions = []

            while 1 <= row <= self.board_size and 1 <= col <= self.board_size:
                if (row, col) not in state.board:
                    break
                elif state.board[(row, col)] == opponent:
                    temp_positions.append((row, col))
                elif state.board[(row, col)] == player:
                    if temp_positions:
                        positions_to_flip.extend(temp_positions)
                    break
                else:
                    break
                row += delta_row
                col += delta_col

        return positions_to_flip

    @override
    def is_terminal(self, state: OthelloState) -> bool:
        return state.is_terminal

    @override
    def reward(self, state: OthelloState, player_id: TPlayerId) -> float:
        if not self.is_terminal(state):
            return 0.0
        player_count = sum(1 for p in state.board.values() if p == player_id)
        opponent_count = sum(1 for p in state.board.values() if p == 3 - player_id)
        if player_count > opponent_count:
            return 1.0
        elif player_count < opponent_count:
            return -1.0
        else:
            return 0.0  # Draw

    @override
    def pretty_str(self, state: OthelloState) -> str:
        def cell_to_str(row: int, col: int) -> str:
            pos = (row, col)
            if pos in state.board:
                return "●" if state.board[pos] == 1 else "○"
            return "."

        rows = []
        for row in range(self.board_size, 0, -1):
            row_str = " ".join(cell_to_str(row, col) for col in range(1, self.board_size + 1))
            rows.append(row_str)

        return "\n".join(rows)

    def parse_board(self, board_str: str, current_player: TPlayerId, is_terminal: bool) -> OthelloState:
        """Parses a board string into an OthelloState."""
        board: Map[TPosition, int] = Map()
        rows = board_str.strip().split("\n")[::-1]
        for r, row in enumerate(rows, start=1):
            cells = row.strip().split()
            for c, cell in enumerate(cells, start=1):
                if cell == "●":
                    board = board.set((r, c), 1)
                elif cell == "○":
                    board = board.set((r, c), 2)
        return OthelloState(board=board, current_player=current_player, is_terminal=is_terminal)


class OthelloSerializer(GameSerializer[OthelloGame, OthelloState, TAction]):
    @override
    def serialize_state(self, game: OthelloGame, state: OthelloState) -> dict[str, Any]:
        """Serialize the game state to a dictionary for frontend consumption."""
        board_size = game.board_size
        # Note: We're not changing the indexing here because the game logic already uses 1-based indexing
        board = [
            [state.board.get((row, col), 0) for col in range(1, board_size + 1)] for row in range(1, board_size + 1)
        ]
        return {
            "rows": board_size,
            "columns": board_size,
            "state": board,
            "current_player": state.current_player,
            "legal_actions": game.legal_actions(state),
            "is_terminal": game.is_terminal(state),
        }

    @override
    def parse_action(self, game: OthelloGame, action_data: dict[str, Any]) -> TAction:
        """Parse an action from frontend data."""
        row = action_data.get("row")
        col = action_data.get("col")
        if row is None or col is None:
            raise ValueError("Action data must include 'row' and 'col'")
        return (int(row), int(col))  # Ensure we're returning integers

    @override
    def state_to_tensor(self, game: OthelloGame, state: OthelloState) -> torch.Tensor:
        board_array = torch.tensor([[state.board.get((row, col), 0) for col in range(1, 9)] for row in range(1, 9)])
        return torch.cat([board_array.flatten(), torch.tensor([state.current_player])])

    @override
    def action_to_tensor(self, game: OthelloGame, action: TAction) -> torch.Tensor:
        return torch.tensor(action)

    @override
    def tensor_to_action(self, game: OthelloGame, action_tensor: torch.Tensor) -> TAction:
        return (int(action_tensor[0].item()), int(action_tensor[1].item()))

    @override
    def tensor_to_state(self, game: OthelloGame, state_tensor: torch.Tensor) -> OthelloState:
        board_array = state_tensor[:-1].reshape(8, 8)
        board = {
            (row + 1, col + 1): int(board_array[row, col].item())
            for row in range(8)
            for col in range(8)
            if board_array[row, col] != 0
        }
        current_player = int(state_tensor[-1].item())
        return OthelloState(board=Map(board), current_player=current_player, is_terminal=False)
