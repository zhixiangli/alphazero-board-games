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
        # Board view (bottom-up idea):
        # col0: B,B,B,.,.,.   col1: W,W,W,.,.,.
        # Expected output: action {14} -> row=2,col=0.
        (
            "B[50];W[51];B[40];W[41];B[30];W[42]",
            ChessType.BLACK,
            {2 * 7 + 0},
        ),
        # Case 2 (input): BLACK can win vertically in column 3.
        # Board view:
        # col3: B,B,B,.,.,.   and filler stones in columns 0/1/2.
        # Expected output: action {17} -> row=2,col=3.
        (
            "B[53];W[50];B[43];W[51];B[33];W[52]",
            ChessType.BLACK,
            {2 * 7 + 3},
        ),
        # Case 3 (input): BLACK has three in a horizontal chain on bottom row.
        # Board view bottom row: B B B . . . .
        # Expected output: action {38} -> row=5,col=3 to complete 4-in-a-row.
        (
            "B[50];W[40];B[51];W[41];B[52];W[42]",
            ChessType.BLACK,
            {5 * 7 + 3},
        ),
        # Case 4 (input): WHITE threatens vertical win in column 0.
        # Expected output: BLACK blocks with action {14} -> row=2,col=0.
        (
            "W[50];B[51];W[40];B[41];W[30];B[42]",
            ChessType.BLACK,
            {2 * 7 + 0},
        ),
        # Case 5 (input): BLACK can complete a diagonal tactical line.
        # Expected output: action {39} -> row=5,col=4.
        (
            "B[50];W[51];B[41];W[52];B[32];W[42];B[40];W[53];B[31];W[43];B[22]",
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
