from __future__ import annotations

import numpy as np

from mcts import State

winner_mask = np.array(
    [
        [1, 1, 1, 0, 0, 0, 0, 0, 0],  # Rows
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 0, 0],  # Columns
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],  # Diagonals
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
    ]
)


class TicTacToe(State):

    def __init__(
        self, board: np.ndarray = np.zeros((2, 3, 3)), current_player: int = 0
    ):
        super().__init__()
        self.board = board.reshape(2, 3, 3)
        self.current_player = current_player
        self.actions = self.legal_actions()

    def render(self) -> None:
        board = self.board.reshape(2, 9)
        print("Current Player:", self.current_player)
        for i in range(3):
            print("-------------")
            out = "| "
            for j in range(3):
                if board[0, i * 3 + j] == 1:
                    piece = "X"
                elif board[1, i * 3 + j] == 1:
                    piece = "O"
                else:
                    piece = " "
                out += piece + " | "
            print(out)
        print("-------------")

    def is_terminal(self) -> bool:
        return self.winner() != -1

    def legal_actions(self) -> list:
        board = self.board.reshape(2, 9)
        return [divmod(i, 3) for i in range(9)
                if board[0, i] == 0 and board[1, i] == 0]

    def step(self, action) -> TicTacToe:
        new_board = np.copy(self.board)
        new_board = new_board.reshape(2, 9)
        new_board[self.current_player, action[0] * 3 + action[1]] = 1
        return TicTacToe(new_board, 1 - self.current_player)

    def reward(self, player: int) -> int:
        winner = self.winner()
        if winner == player:
            return -1
        if winner == 1 - player:
            return 1
        return 0

    def winner(self) -> int:
        board = self.board.reshape(2, 9)
        for player in range(2):
            for mask in winner_mask:
                if np.all(board[player][mask == 1] == 1):
                    return player
        if (len(self.actions) == 0):
            return 2
        return -1
