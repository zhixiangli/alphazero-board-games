#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import pytest

from alphazero.tests.end_to_end_test_utils import (
    assert_best_actions_from_checkpoint_in_expected_set,
)
from gomoku_15_15.config import GomokuConfig
from gomoku_15_15.game import ChessType, GomokuGame


@pytest.mark.integration
@pytest.mark.parametrize(
    "board,player,expected_best_actions",
    [
        (
            "B[77];W[00];B[78];W[01];B[79];W[02];B[7a];W[03]",
            ChessType.BLACK,
            {7 * 15 + 6, 7 * 15 + 11},
        ),
        (
            "B[77];W[00];B[87];W[01];B[97];W[02];B[a7];W[03]",
            ChessType.BLACK,
            {6 * 15 + 7, 11 * 15 + 7},
        ),
        (
            "B[66];W[00];B[77];W[01];B[88];W[02];B[99];W[03]",
            ChessType.BLACK,
            {5 * 15 + 5, 10 * 15 + 10},
        ),
        (
            "B[6a];W[00];B[79];W[01];B[88];W[02];B[97];W[03]",
            ChessType.BLACK,
            {5 * 15 + 11, 10 * 15 + 6},
        ),
        (
            "W[77];B[00];W[78];B[01];W[79];B[02];W[7a];B[03]",
            ChessType.BLACK,
            {0 * 15 + 4},
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
        simulation_num=160,
        checkpoint_dir="gomoku_15_15/data",
    )
