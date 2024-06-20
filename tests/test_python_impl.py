import pytest
import numpy as np

from mcts.python_impl import MCTS as MCTS_py, Node as Node_py
from .tictactoe import TicTacToe

initial_state = TicTacToe()


def test_node():
    node = Node_py(initial_state, to_play=0)
    assert node.state == initial_state
    assert node.action == (-1, -1)
    assert node.to_play == 0
    assert node.parent is None
    assert node.reward_sum == 0.0
    assert node.N == 0
    assert node.Q == 0.0
    assert node.children == []

def test_mcts():
    node = Node_py(initial_state, to_play=0)
    mcts = MCTS_py(node, 1.0, 25)
    assert mcts.root == node
    assert mcts.C == 1.0
    assert mcts.num_simulations == 25

def test_mcts_search():
    node = Node_py(initial_state, to_play=0)
    mcts = MCTS_py(node, 1.0, 25)
    action = mcts.search()
    assert action in initial_state.legal_actions()


def test_winning_move_X():
    state = TicTacToe()
    state.board = np.array([[[1, 1, 0], [0, 0, 0], [0, 0, 0]],
                            [[0, 0, 0], [1, 1, 0], [0, 0, 0]]])
    state.current_player = 0
    node = Node_py(state, to_play=0)
    mcts = MCTS_py(node, 1.0, 100)
    action = mcts.search()
    assert action == (0, 2)

def test_winning_move_O():
    state = TicTacToe()
    state.board = np.array([[[0, 1, 1], [1, 0, 0], [0, 0, 0]],
                            [[1, 0, 0], [0, 1, 0], [0, 0, 0]]])
    state.current_player = 1
    node = Node_py(state, to_play=1)
    mcts = MCTS_py(node, 1.0, 100)
    action = mcts.search()
    assert action == (2, 2)

def test_block_winning_move_X():
    state = TicTacToe()
    state.board = np.array([[[0, 0, 1], [0, 1, 0], [0, 0, 0]],
                            [[1, 0, 0], [1, 0, 0], [0, 0, 0]]])
    state.current_player = 0
    node = Node_py(state, to_play=0)
    mcts = MCTS_py(node, 1.0, 100)
    action = mcts.search()
    assert action == (2, 0)

def test_block_winning_move_O():
    state = TicTacToe()
    state.board = np.array([[[1, 0, 0], [1, 0, 0], [0, 0, 0]],
                            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    state.current_player = 1
    node = Node_py(state, to_play=1)
    mcts = MCTS_py(node, 1.0, 100)
    action = mcts.search()
    assert action == (2, 0)
