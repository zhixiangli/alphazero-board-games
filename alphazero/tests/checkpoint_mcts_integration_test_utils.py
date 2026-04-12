#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import glob
import os

import numpy

from alphazero.mcts import MCTS
from alphazero.nnet import AlphaZeroNNet


def assert_best_actions_from_checkpoint_in_expected_set(
    config,
    game,
    board,
    player,
    expected_best_actions,
    simulation_num,
    checkpoint_dir,
):
    config.simulation_num = simulation_num

    checkpoints = glob.glob(config.save_checkpoint_path + "*.pt")
    assert checkpoints, f"No checkpoints found for pattern: {config.save_checkpoint_path}*.pt"

    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    assert latest_checkpoint.startswith(f"./{checkpoint_dir}/model.")

    nnet = AlphaZeroNNet(game, config)
    nnet.load_checkpoint(config.save_checkpoint_path)

    mcts = MCTS(nnet, game, config)
    actions, counts = mcts.simulate(board, player)
    if len(actions) == 0:
        assert False, "MCTS unexpectedly returned no actions"

    best = numpy.max(counts)
    best_actions = actions[counts == best]
    assert set(best_actions.tolist()).issubset(expected_best_actions)
