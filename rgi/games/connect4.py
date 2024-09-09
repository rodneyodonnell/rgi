from rgi.core.game import Game

class Connect4(Game[list[list[int]], int, int]):
    def __init__(self, rows: int = 6, cols: int = 7):
        self.rows = rows
        self.cols = cols

    def initial_state(self) -> list[list[int]]:
        return [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def current_player(self, state: list[list[int]]) -> int:
        return 1 if sum(sum(row) for row in state) % 2 == 0 else 2

    def legal_actions(self, state: list[list[int]]) -> list[int]:
        return [col for col in range(self.cols) if state[0][col] == 0]

    def next_state(self, state: list[list[int]], action: int) -> list[list[int]]:
        new_state = [row.copy() for row in state]
        for row in range(self.rows - 1, -1, -1):
            if new_state[row][action] == 0:
                new_state[row][action] = self.current_player(state)
                break
        return new_state

    def is_terminal(self, state: list[list[int]]) -> bool:
        return self._check_win(state, 1) or self._check_win(state, 2) or len(self.legal_actions(state)) == 0

    def reward(self, state: list[list[int]], player: int) -> float:
        if self._check_win(state, player):
            return 1.0
        elif self._check_win(state, 3 - player):  # 3 - player gives the opponent
            return -1.0
        else:
            return 0.0

    def action_to_string(self, action: int) -> str:
        return str(action)

    def state_to_string(self, state: list[list[int]]) -> str:
        return "\n".join(" ".join(str(cell) for cell in row) for row in state)

    def string_to_action(self, string: str) -> int:
        return int(string)

    def _check_win(self, state: list[list[int]], player: int) -> bool:
        # Check horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(state[row][col+i] == player for i in range(4)):
                    return True

        # Check vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if all(state[row+i][col] == player for i in range(4)):
                    return True

        # Check diagonal (top-left to bottom-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(state[row+i][col+i] == player for i in range(4)):
                    return True

        # Check diagonal (top-right to bottom-left)
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                if all(state[row+i][col-i] == player for i in range(4)):
                    return True

        return False