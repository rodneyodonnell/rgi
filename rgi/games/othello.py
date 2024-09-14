from typing import Optional
from typing_extensions import override
from immutables import Map
from rgi.core.base import Game, TPlayerId, TAction


class OthelloState:
    def __init__(self, board: Map[tuple[int, int], int], current_player: int, is_terminal: bool):
        # Immutable map of board positions to player IDs (1 or 2)
        self.board = board
        self.current_player = current_player
        self.is_terminal = is_terminal

    def __repr__(self) -> str:
        return f"OthelloState(board={self.board}, current_player={self.current_player}, is_teminal={self.is_terminal})"


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
        board = Map()
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

    @override
    def all_player_ids(self, state: OthelloState) -> list[TPlayerId]:
        return [1, 2]

    @override
    def legal_actions(self, state: OthelloState) -> list[tuple[int, int]]:
        return self._get_legal_moves(state, state.current_player)

    def _get_legal_moves(self, state: OthelloState, player: int) -> list[tuple[int, int]]:
        opponent = 3 - player
        legal_moves = []

        for position in self._all_positions:
            if position in state.board:
                continue  # Skip occupied positions
            if self._would_flip(state, position, player, opponent):
                legal_moves.append(position)
        return legal_moves

    def _would_flip(self, state: OthelloState, position: tuple[int, int], player: int, opponent: int) -> bool:
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
    def next_state(self, state: OthelloState, action: tuple[int, int]) -> OthelloState:
        _legal_actions = self.legal_actions(state)
        if action not in _legal_actions:
            raise ValueError(f"Invalid move: {action} is not a legal action {_legal_actions}.")

        player = state.current_player
        opponent = 3 - player
        new_board = state.board

        positions_to_flip = self._get_positions_to_flip(state, action, player, opponent)

        # Place the new disc
        new_board = new_board.set(action, player)

        # Flip the opponent's discs
        for pos in positions_to_flip:
            new_board = new_board.set(pos, player)

        # Determine next player
        next_player = 3 - player  # Switch turns

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
        self, state: OthelloState, position: tuple[int, int], player: int, opponent: int
    ) -> list[tuple[int, int]]:
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
    def reward(self, state: OthelloState, player_id: int) -> float:
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
        board_str = ""
        for row in reversed(range(1, self.board_size + 1)):
            row_str = ""
            for col in range(1, self.board_size + 1):
                pos = (row, col)
                if pos in state.board:
                    player = state.board[pos]
                    row_str += " ●" if player == 1 else " ○"
                else:
                    row_str += " ."
            board_str += row_str + "\n"
        return board_str

    def parse_board(self, board_str: str, current_player: int, is_terminal: bool) -> OthelloState:
        """Parses a board string into an OthelloState."""
        board = Map()
        rows = board_str.strip().split("\n")[::-1]
        for r, row in enumerate(rows, start=1):
            cells = row.strip().split()
            for c, cell in enumerate(cells, start=1):
                if cell == "●":
                    board = board.set((r, c), 1)
                elif cell == "○":
                    board = board.set((r, c), 2)
        return OthelloState(board=board, current_player=current_player, is_terminal=is_terminal)
