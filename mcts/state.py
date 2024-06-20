from __future__ import annotations
from abc import abstractmethod


class State:

    def __init__(self):
        self.current_player = 0
        self.board = None

    @abstractmethod
    def render(self) -> None:
        """Render the current state of the game."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the game is over."""
        pass

    @abstractmethod
    def legal_actions(self) -> list:
        """Return a list of legal actions."""
        pass

    @abstractmethod
    def step(self, action) -> State:
        """Return the new state after taking the given action."""
        pass

    @abstractmethod
    def reward(self, player: int) -> int:
        """
        Return the reward for the given player of a terminal state.
        Note: This reward is the perspective of the other player.
        Return 1 if the other player wins, -1 if the other player loses,
        and 0 for a draw.
        """
        pass
