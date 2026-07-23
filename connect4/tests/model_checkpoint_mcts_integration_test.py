#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import pytest

from alphazero.tests.checkpoint_mcts_integration_test_utils import (
    assert_best_actions_from_checkpoint_in_expected_set,
)
from connect4.config import Connect4Config
from connect4.game import ChessType, Connect4Game


@pytest.mark.integration
@pytest.mark.parametrize(
    "board,player,expected_best_actions",
    [
        # Case 1 (input): BLACK can win vertically in column 0.
        # Board (row 0 at top):
        #   0 1 2 3 4 5 6
        # 0 . . . . . . .
        # 1 . . . . . . .
        # 2 . . . . . . .
        # 3 B . . . . . .
        # 4 B W . . . . .
        # 5 B W W . . . .
        # Expected output: action {14} -> row=2,col=0.
        (
            "B[5,0];W[5,1];B[4,0];W[4,1];B[3,0];W[4,2]",
            ChessType.BLACK,
            {2 * 7 + 0},
        ),
        # Case 2 (input): BLACK can win vertically in column 3.
        # Board (row 0 at top):
        #   0 1 2 3 4 5 6
        # 0 . . . . . . .
        # 1 . . . . . . .
        # 2 . . . . . . .
        # 3 . . . B . . .
        # 4 . . . B . . .
        # 5 W W W B . . .
        # Expected output: action {17} -> row=2,col=3.
        (
            "B[5,3];W[5,0];B[4,3];W[5,1];B[3,3];W[5,2]",
            ChessType.BLACK,
            {2 * 7 + 3},
        ),
        # Case 3 (input): BLACK has three in a horizontal chain on bottom row.
        # Board (row 0 at top):
        #   0 1 2 3 4 5 6
        # 0 . . . . . . .
        # 1 . . . . . . .
        # 2 . . . . . . .
        # 3 . . . . . . .
        # 4 W W W . . . .
        # 5 B B B . . . .
        # Expected output: action {38} -> row=5,col=3 to complete 4-in-a-row.
        (
            "B[5,0];W[4,0];B[5,1];W[4,1];B[5,2];W[4,2]",
            ChessType.BLACK,
            {5 * 7 + 3},
        ),
        # Case 4 (input): WHITE threatens vertical win in column 0.
        # Board (row 0 at top):
        #   0 1 2 3 4 5 6
        # 0 . . . . . . .
        # 1 . . . . . . .
        # 2 . . . . . . .
        # 3 W . . . . . .
        # 4 W B . . . . .
        # 5 W B B . . . .
        # Expected output: BLACK blocks with action {14} -> row=2,col=0.
        (
            "W[5,0];B[5,1];W[4,0];B[4,1];W[3,0];B[4,2]",
            ChessType.BLACK,
            {2 * 7 + 0},
        ),
        # Case 5 (input): BLACK can complete a diagonal tactical line.
        # Board (row 0 at top):
        #   0 1 2 3 4 5 6
        # 0 . . . . . . .
        # 1 . . . . . . .
        # 2 . . B . . . .
        # 3 . B B W . . .
        # 4 B B W W . . .
        # 5 B W W W . . .
        # Expected output: action {39} -> row=5,col=4.
        (
            "B[5,0];W[5,1];B[4,1];W[5,2];B[3,2];W[4,2];B[4,0];W[5,3];B[3,1];W[4,3];B[2,2]",
            ChessType.BLACK,
            {5 * 7 + 4},
        ),
    ],
    ids=[
        "vertical_win_col0",
        "vertical_win_col3",
        "horizontal_win",
        "block_opponent_vertical",
        "diagonal_win",
    ],
)
def test_model_checkpoint_and_mcts_select_expected_tactical_actions(
    board, player, expected_best_actions
):
    config = Connect4Config()
    game = Connect4Game(config)

    assert_best_actions_from_checkpoint_in_expected_set(
        config=config,
        game=game,
        board=board,
        player=player,
        expected_best_actions=expected_best_actions,
        simulation_num=120,
        checkpoint_dir="connect4/data",
    )
