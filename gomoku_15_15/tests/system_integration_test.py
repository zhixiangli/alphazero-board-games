#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pytest

from alphazero.mcts import MCTS
from alphazero.nnet import AlphaZeroNNet
from gomoku_15_15.config import GomokuConfig
from gomoku_15_15.game import ChessType, GomokuGame


def _build_sgf(moves):
    return ";".join(f"{player}[{row:x}{col:x}]" for player, row, col in moves)


def _best_action(mcts, board, player):
    actions, counts = mcts.simulate(board, player)
    return int(actions[counts.argmax()])


@pytest.fixture(scope="module")
def gomoku15_system():
    config = GomokuConfig()
    config.simulation_num = 300

    game = GomokuGame(config)
    nnet = AlphaZeroNNet(game, config)
    nnet.load_checkpoint(config.save_checkpoint_path)
    return game, nnet, config


@pytest.mark.integration
@pytest.mark.parametrize(
    "name,moves,player,expected_actions",
    [
        (
            "open_three",
            [
                # Row 8 visualization (0-index row 7):
                #   ... . B B B . ...
                # Best tactical extension for BLACK is either side of the chain.
                (ChessType.BLACK, 7, 7),
                (ChessType.BLACK, 7, 8),
                (ChessType.BLACK, 7, 9),
                (ChessType.WHITE, 0, 0),
                (ChessType.WHITE, 1, 1),
            ],
            ChessType.BLACK,
            {7 * 15 + 6, 7 * 15 + 10},
        ),
        (
            "open_four",
            [
                # Row 8 visualization (0-index row 7):
                #   ... . B B B B . ...
                # BLACK can win immediately by playing at either open end.
                (ChessType.BLACK, 7, 6),
                (ChessType.BLACK, 7, 7),
                (ChessType.BLACK, 7, 8),
                (ChessType.BLACK, 7, 9),
                (ChessType.WHITE, 0, 0),
                (ChessType.WHITE, 0, 1),
            ],
            ChessType.BLACK,
            {7 * 15 + 5, 7 * 15 + 10},
        ),
        (
            "closed_four",
            [
                # Row 8 visualization (0-index row 7):
                #   ... W B B B B . ...
                # Left side is blocked by WHITE, so only the right end wins.
                (ChessType.BLACK, 7, 7),
                (ChessType.BLACK, 7, 8),
                (ChessType.BLACK, 7, 9),
                (ChessType.BLACK, 7, 10),
                (ChessType.WHITE, 7, 6),
            ],
            ChessType.BLACK,
            {7 * 15 + 11},
        ),
        (
            "block_opponent_open_four",
            [
                # Row 8 visualization (0-index row 7):
                #   ... . W W W W . ...
                # BLACK must block one of WHITE's two winning endpoints.
                (ChessType.WHITE, 7, 6),
                (ChessType.WHITE, 7, 7),
                (ChessType.WHITE, 7, 8),
                (ChessType.WHITE, 7, 9),
                (ChessType.BLACK, 0, 0),
            ],
            ChessType.BLACK,
            {7 * 15 + 5, 7 * 15 + 10},
        ),
    ],
    ids=lambda name: name,
)
def test_system_selects_expected_tactical_move(
    gomoku15_system, name, moves, player, expected_actions
):
    game, nnet, config = gomoku15_system
    board = _build_sgf(moves)
    mcts = MCTS(nnet, game, config)

    best_action = _best_action(mcts, board, player)

    assert best_action in expected_actions, (
        f"{name}: best action {best_action} not in expected {sorted(expected_actions)}"
    )
