#!/usr/bin/python3
#  -*- coding: utf-8 -*-


import copy
import itertools
import logging
import os
import pickle
import random
import threading
import tempfile
from collections import deque

import numpy

from alphazero.mcts import MCTS


class RL:
    def __init__(self, nnet, game, args):
        self.nnet = nnet
        self.game = game
        self.args = args
        self.sample_pool = deque(maxlen=args.max_sample_pool_size)
        self._persistence_condition = threading.Condition()
        self._pending_sample_pool = None
        self._persistence_thread = None

        persisted_sample_pool = self.read_sample_pool()
        if persisted_sample_pool:
            self.sample_pool.extend(persisted_sample_pool)
        logging.info(
            "samples currsize: %d, maxsize: %d",
            len(self.sample_pool),
            self.sample_pool.maxlen,
        )

    def play_against_itself(self):
        board, player = self.game.get_initial_state()
        canonical_boards, players, policies = [], [], []
        mcts = MCTS(self.nnet, self.game, self.args)
        max_moves = self.args.rows * self.args.columns
        for i in itertools.count():
            actions, counts = mcts.simulate(
                board, player, add_root_noise=i < self.args.temp_step
            )
            pi = counts / numpy.sum(counts)
            policy = numpy.zeros(self.args.rows * self.args.columns)
            policy[actions] = pi
            canonical_boards.append(self.game.get_canonical_form(board, player))
            players.append(player)
            policies.append(policy)

            if i >= self.args.temp_step:
                # After the stochastic opening window, follow MCTS visit counts
                # directly (greedy) to avoid noise-driven placements.
                action = actions[numpy.argmax(pi)]
            else:
                action = numpy.random.choice(actions, p=pi)

            next_board, next_player = self.game.next_state(board, action, player)
            winner = self.game.is_terminal_state(next_board, action, player)
            if winner is not None:
                logging.info("winner: %s", winner)
                values = numpy.array(
                    [self.game.compute_reward(winner, p) for p in players]
                )
                return [i for i in zip(canonical_boards, policies, values)]
            assert i < max_moves, (
                "Game exceeded maximum possible moves (%d). "
                "Terminal state detection may be broken." % max_moves
            )
            board, player = next_board, next_player

    def start(self):
        for i in itertools.count():
            logging.info("iteration %d:", i)
            samples = self.play_against_itself()
            augmented_data = self.game.augment_samples(samples)
            self.sample_pool.extend(augmented_data)
            logging.info(
                "augmented_data len: %d, current sample pool size: %d",
                len(augmented_data),
                len(self.sample_pool),
            )
            if (
                self.args.batch_size <= len(self.sample_pool)
                and (i + 1) % self.args.train_interval == 0
            ):
                self.nnet.train(random.sample(self.sample_pool, self.args.batch_size))
                self.queue_sample_pool_persistence(self.sample_pool)
                self.nnet.save_checkpoint(self.args.save_checkpoint_path)

    def persist_sample_pool(self, samples):
        """Synchronously persist *samples* for callers that need durability now."""
        self._write_sample_pool(samples)

    def queue_sample_pool_persistence(self, samples):
        """Persist only the newest snapshot without accumulating writer threads."""
        with self._persistence_condition:
            self._pending_sample_pool = copy.deepcopy(samples)
            if self._persistence_thread is None:
                self._persistence_thread = threading.Thread(
                    target=self._persist_pending_sample_pools,
                    name="sample-pool-persistence",
                )
                self._persistence_thread.start()
            self._persistence_condition.notify()

    def wait_for_persistence(self):
        """Wait until the queued replay-pool snapshot has been written."""
        while True:
            with self._persistence_condition:
                persistence_thread = self._persistence_thread
            if persistence_thread is None:
                return
            persistence_thread.join()

    def _persist_pending_sample_pools(self):
        while True:
            with self._persistence_condition:
                samples = self._pending_sample_pool
                self._pending_sample_pool = None
            self._write_sample_pool(samples)
            with self._persistence_condition:
                if self._pending_sample_pool is None:
                    self._persistence_thread = None
                    self._persistence_condition.notify_all()
                    return

    def _write_sample_pool(self, samples):
        logging.info("persist sample pool start")
        directory = os.path.dirname(self.args.sample_pool_file) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=".samples-", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(samples, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.args.sample_pool_file)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise
        logging.info("persist sample pool done")

    def read_sample_pool(self):
        if not os.path.exists(self.args.sample_pool_file):
            return None
        try:
            with open(self.args.sample_pool_file, "rb") as f:
                logging.info("load samples from %s", self.args.sample_pool_file)
                return pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError) as e:
            logging.warning(
                "ignoring unreadable sample pool at %s: %s",
                self.args.sample_pool_file,
                e,
            )
            return None
