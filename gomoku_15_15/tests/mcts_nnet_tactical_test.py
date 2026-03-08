#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import numpy
import pytest

from alphazero.mcts import MCTS
from alphazero.nnet import NNet
from gomoku_15_15.game import ChessType, GomokuGame


class _TacticalPatternNNet(NNet):
    def __init__(self, rows, columns, n_in_row):
        self.rows = rows
        self.columns = columns
        self.n_in_row = n_in_row
        self.directions = ((1, 1), (1, -1), (0, 1), (1, 0))

    def predict(self, board):
        current = board[:, :, 0].astype(bool)
        opponent = board[:, :, 1].astype(bool)
        available = ~(current | opponent)

        tactical_moves = self._find_winning_moves(current, available)
        if not tactical_moves:
            tactical_moves = self._find_winning_moves(opponent, available)

        policy = numpy.zeros(self.rows * self.columns)
        if tactical_moves:
            policy[tactical_moves] = 1.0 / len(tactical_moves)
        else:
            policy[numpy.flatnonzero(available.reshape(-1))] = 1.0
        return policy, 0.0

    def _find_winning_moves(self, stones, available):
        moves = []
        for action in numpy.flatnonzero(available.reshape(-1)):
            row, col = divmod(int(action), self.columns)
            if self._is_winning_move(stones, row, col):
                moves.append(int(action))
        return moves

    def _is_winning_move(self, stones, row, col):
        for dx, dy in self.directions:
            count = 1
            for step in (1, -1):
                x, y = row + step * dx, col + step * dy
                while 0 <= x < self.rows and 0 <= y < self.columns and stones[x, y]:
                    count += 1
                    x += step * dx
                    y += step * dy
            if count >= self.n_in_row:
                return True
        return False


@pytest.fixture
def gomoku15_tactical_game(make_args):
    return GomokuGame(make_args(rows=15, columns=15, n_in_row=5))


@pytest.fixture
def tactical_mcts_args(make_args):
    return make_args(rows=15, columns=15, n_in_row=5, simulation_num=32, c_puct=5)


@pytest.fixture
def tactical_nnet():
    return _TacticalPatternNNet(rows=15, columns=15, n_in_row=5)


@pytest.mark.integration
@pytest.mark.parametrize(
    "moves,player,expected_action",
    [
        (
            [
                (ChessType.WHITE, 7, 9),
                (ChessType.BLACK, 7, 10),
                (ChessType.BLACK, 7, 11),
                (ChessType.BLACK, 7, 12),
                (ChessType.BLACK, 7, 13),
                (ChessType.WHITE, 0, 0),
            ],
            ChessType.BLACK,
            7 * 15 + 14,
        ),
        (
            [
                (ChessType.BLACK, 8, 3),
                (ChessType.BLACK, 8, 4),
                (ChessType.BLACK, 8, 6),
                (ChessType.BLACK, 8, 7),
                (ChessType.WHITE, 8, 2),
                (ChessType.WHITE, 0, 0),
            ],
            ChessType.BLACK,
            8 * 15 + 5,
        ),
        (
            [
                (ChessType.WHITE, 10, 10),
                (ChessType.WHITE, 10, 11),
                (ChessType.WHITE, 10, 12),
                (ChessType.WHITE, 10, 13),
                (ChessType.BLACK, 10, 9),
                (ChessType.BLACK, 0, 0),
            ],
            ChessType.BLACK,
            10 * 15 + 14,
        ),
        (
            [
                (ChessType.WHITE, 9, 9),
                (ChessType.BLACK, 10, 10),
                (ChessType.BLACK, 11, 11),
                (ChessType.BLACK, 12, 12),
                (ChessType.BLACK, 13, 13),
                (ChessType.WHITE, 0, 0),
            ],
            ChessType.BLACK,
            14 * 15 + 14,
        ),
    ],
    ids=[
        "complete_horizontal_four",
        "fill_broken_four_gap",
        "block_opponent_four",
        "complete_diagonal_four",
    ],
)
def test_mcts_nnet_pipeline_selects_expected_tactical_move(
    gomoku15_tactical_game,
    tactical_mcts_args,
    tactical_nnet,
    build_sgf,
    moves,
    player,
    expected_action,
):
    board = build_sgf(moves)

    actions, counts = MCTS(tactical_nnet, gomoku15_tactical_game, tactical_mcts_args).simulate(
        board, player
    )

    assert list(actions)[int(numpy.argmax(counts))] == expected_action
