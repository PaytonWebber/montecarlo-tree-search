import pytest
from mcts.python_impl import MCTS as MCTS_py, Node as Node_py
from .tictactoe import TicTacToe

initial_state = TicTacToe()


def test_node():
    node = Node_py(initial_state, (0, 0), 0, None)
    assert node.state == initial_state
    assert node.action == (0, 0)
    assert node.to_play == 0
    assert node.parent is None
    assert node.reward_sum == 0.0
    assert node.N == 0
    assert node.Q == 0.0
    assert node.children == []


def test_mcts():
    node = Node_py(initial_state, (0, 0), 0, None)
    mcts = MCTS_py(node, 1.0, 100)
    assert mcts.root == node
    assert mcts.C == 1.0
    assert mcts.num_simulations == 100


def test_mcts_search():
    node = Node_py(initial_state, (0, 0), 0, None)
    mcts = MCTS_py(node, 1.0, 100)
    action = mcts.search()
    assert action in initial_state.legal_actions()
