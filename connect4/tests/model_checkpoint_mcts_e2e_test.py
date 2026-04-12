#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import pytest

from alphazero.tests.end_to_end_test_utils import (
    assert_best_actions_from_checkpoint_in_expected_set,
)
from connect4.config import Connect4Config
from connect4.game import ChessType, Connect4Game


@pytest.mark.integration
@pytest.mark.parametrize(
    "board,player,expected_best_actions",
    [
        (
            "B[50];W[51];B[40];W[41];B[30];W[42]",
            ChessType.BLACK,
            {2 * 7 + 0},
        ),
        (
            "B[53];W[50];B[43];W[51];B[33];W[52]",
            ChessType.BLACK,
            {2 * 7 + 3},
        ),
        (
            "B[50];W[40];B[51];W[41];B[52];W[42]",
            ChessType.BLACK,
            {5 * 7 + 3},
        ),
        (
            "W[50];B[51];W[40];B[41];W[30];B[42]",
            ChessType.BLACK,
            {2 * 7 + 0},
        ),
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
