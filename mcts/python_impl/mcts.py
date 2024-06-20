from __future__ import annotations
from typing import Optional

import numpy as np


class Node:

    def __init__(
            self,
            state,
            to_play: int = 0,
            action: tuple[int, int] = (-1, -1),
            parent: Optional[Node] = None
    ):
        self.state = state
        self.action = action
        self.to_play = to_play
        self.parent = parent
        self.reward_sum: float = 0.0
        self.N: int = 0  # Number of visits
        self.Q: float = 0.0  # Average reward
        self.children: list[Node] = list()

    def __str__(self) -> str:
        return f"Node: {self.action} | N: {self.N} | Q: {self.Q}"

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node (i.e. no children)."""
        return len(self.children) == 0

    def _exploitation_value(self) -> float:
        """The exploitation value of the node."""
        return self.Q

    def _exploration_value(self, C: float) -> float:
        """The exploration value of the node."""
        if self.parent is None:
            return 0  # For root node to make typechecker happy
        return C * np.sqrt(np.log(self.parent.N + 1e-5) / (self.N + 1))

    def UCB(self, C: float) -> float:
        """Upper Confidence Bound (UCB) value of the node."""
        if self.parent is None:
            return 0  # For root node to make typechecker happy
        return self._exploitation_value() + self._exploration_value(C)


class MCTS:

    def __init__(self, root: Node, C: float, num_simulations: int):
        self.root = root
        self.C = C
        self.num_simulations = num_simulations

    def search(self) -> tuple:
        """Search for the best action to take from the current state."""
        for _ in range(self.num_simulations):
            node = self._select()
            if not node.state.is_terminal() and (node.is_leaf() and node.N > 0):
                self._expand(node)
                node = node.children[0]
            reward = self._playout(node)
            self._backpropagate(node, reward)
        return max(self.root.children, key=lambda n: n.N).action

    def _select(self) -> Node:
        """Selecting the best node to explore next."""
        node = self.root
        while not node.is_leaf() and not node.state.is_terminal():
            node = max(node.children, key=lambda n: n.UCB(self.C))
        return node

    def _expand(self, node: Node) -> None:
        """Expanding the node by adding all possible actions as children."""
        for action in node.state.legal_actions():
            new_state = node.state.step(action)
            new_node = Node(new_state, 1 - node.to_play, action, parent=node)
            node.children.append(new_node)

    def _playout(self, node: Node) -> float:
        """Simulating a random playout from the node to a terminal state."""
        state = node.state
        while not state.is_terminal():
            action = state.actions[np.random.choice(len(state.actions))]
            state = state.step(action)
        return state.reward(node.to_play)

    def _backpropagate(self, node: Node, reward: float) -> None:
        """Backpropagating the reward up the tree."""
        while True:
            node.N += 1
            node.reward_sum += reward
            node.Q = node.reward_sum / node.N
            if node.parent is None:
                break
            node = node.parent
            reward = -reward
