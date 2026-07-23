#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import pytest

from alphazero.tests.checkpoint_mcts_integration_test_utils import (
    assert_best_actions_from_checkpoint_in_expected_set,
)
from gomoku_9_9.config import GomokuConfig
from gomoku_9_9.game import ChessType, GomokuGame


@pytest.mark.integration
@pytest.mark.parametrize(
    "board,player,expected_best_actions",
    [
        # Case 1 (input): BLACK has horizontal four at row 4, cols 4-7.
        # 9x9 board (row 0 top, col 0 left):
        # 0 W W W W . . . . .
        # 1 . . . . . . . . .
        # 2 . . . . . . . . .
        # 3 . . . . . . . . .
        # 4 . . . . B B B B .
        # 5 . . . . . . . . .
        # 6 . . . . . . . . .
        # 7 . . . . . . . . .
        # 8 . . . . . . . . .
        # Expected output: winning extension at either end {39, 44}.
        (
            "B[4,4];W[0,0];B[4,5];W[0,1];B[4,6];W[0,2];B[4,7];W[0,3]",
            ChessType.BLACK,
            {4 * 9 + 3, 4 * 9 + 8},
        ),
        # Case 2 (input): BLACK has vertical four at col 4, rows 4-7.
        # 9x9 board:
        # 0 W W W W . . . . .
        # 1 . . . . . . . . .
        # 2 . . . . . . . . .
        # 3 . . . . . . . . .
        # 4 . . . . B . . . .
        # 5 . . . . B . . . .
        # 6 . . . . B . . . .
        # 7 . . . . B . . . .
        # 8 . . . . . . . . .
        # Expected output: winning extension at either end {31, 76}.
        (
            "B[4,4];W[0,0];B[5,4];W[0,1];B[6,4];W[0,2];B[7,4];W[0,3]",
            ChessType.BLACK,
            {3 * 9 + 4, 8 * 9 + 4},
        ),
        # Case 3 (input): BLACK has main-diagonal four (2,2)-(5,5).
        # 9x9 board:
        # 0 W W W W . . . . .
        # 1 . . . . . . . . .
        # 2 . . B . . . . . .
        # 3 . . . B . . . . .
        # 4 . . . . B . . . .
        # 5 . . . . . B . . .
        # 6 . . . . . . . . .
        # 7 . . . . . . . . .
        # 8 . . . . . . . . .
        # Expected output: winning extension at either end {10, 60}.
        (
            "B[2,2];W[0,0];B[3,3];W[0,1];B[4,4];W[0,2];B[5,5];W[0,3]",
            ChessType.BLACK,
            {1 * 9 + 1, 6 * 9 + 6},
        ),
        # Case 4 (input): BLACK has anti-diagonal four (2,6)-(5,3).
        # 9x9 board:
        # 0 W W W W . . . . .
        # 1 . . . . . . . . .
        # 2 . . . . . . B . .
        # 3 . . . . . B . . .
        # 4 . . . . B . . . .
        # 5 . . . B . . . . .
        # 6 . . . . . . . . .
        # 7 . . . . . . . . .
        # 8 . . . . . . . . .
        # Expected output: winning extension at either end {16, 56}.
        (
            "B[2,6];W[0,0];B[3,5];W[0,1];B[4,4];W[0,2];B[5,3];W[0,3]",
            ChessType.BLACK,
            {1 * 9 + 7, 6 * 9 + 2},
        ),
        # Case 5 (input): WHITE has horizontal four threat at row 4.
        # 9x9 board:
        # 0 B B B B . . . . .
        # 1 . . . . . . . . .
        # 2 . . . . . . . . .
        # 3 . . . . . . . . .
        # 4 . . . . W W W W .
        # 5 . . . . . . . . .
        # 6 . . . . . . . . .
        # 7 . . . . . . . . .
        # 8 . . . . . . . . .
        # Expected output: checkpoint+MCTS defensive choice {4} -> row=0,col=4.
        (
            "W[4,4];B[0,0];W[4,5];B[0,1];W[4,6];B[0,2];W[4,7];B[0,3]",
            ChessType.BLACK,
            {0 * 9 + 4},
        ),
    ],
    ids=[
        "horizontal_win",
        "vertical_win",
        "diagonal_win",
        "anti_diagonal_win",
        "defensive_block",
    ],
)
def test_model_checkpoint_and_mcts_select_expected_tactical_actions(
    board, player, expected_best_actions
):
    config = GomokuConfig()
    game = GomokuGame(config)

    assert_best_actions_from_checkpoint_in_expected_set(
        config=config,
        game=game,
        board=board,
        player=player,
        expected_best_actions=expected_best_actions,
        simulation_num=120,
        checkpoint_dir="gomoku_9_9/data",
    )
