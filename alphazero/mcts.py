#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import numpy


class MCTS:
    def __init__(self, nnet, game, args):
        self.nnet = nnet
        self.game = game
        self.args = args

        self.visit_count = {}  # N(s, a) is the visit count
        self.mean_action_value = {}  # Q(s, a) is the mean action value
        self.prior_probability = {}  # P(s, a) is the prior probability of selecting that edge.

        self.terminal_state = {}
        self.total_visit_count = {}
        self.available_actions = {}

    def simulate(self, board, player, add_root_noise=False):
        if self.args.simulation_num < 2:
            raise ValueError("simulation_num must be at least 2")
        if add_root_noise:
            self.__add_root_dirichlet_noise(board, player)
        for _ in range(self.args.simulation_num):
            self.search(board, player)
        state = self._state_key(board, player)
        self.game.log_status(
            board,
            numpy.copy(self.visit_count[state]),
            numpy.copy(self.available_actions[state]),
        )
        return numpy.copy(self.available_actions[state]), numpy.copy(
            self.visit_count[state]
        )

    @staticmethod
    def _state_key(board, player):
        return board, player

    def __add_root_dirichlet_noise(self, board, player):
        state = self._state_key(board, player)
        if state not in self.prior_probability:
            self.__expand(board, player)

        priors = self.prior_probability[state]
        if len(priors) == 0:
            return

        dirichlet_noise = numpy.random.dirichlet(
            self.args.dirichlet_alpha * numpy.ones(len(priors))
        )
        self.prior_probability[state] = (
            (1.0 - self.args.dirichlet_epsilon) * priors
            + self.args.dirichlet_epsilon * dirichlet_noise
        )

    def search(self, board, player):
        state = self._state_key(board, player)
        if state not in self.prior_probability:  # leaf
            return -self.__expand(board, player)
        index = self.__select(state)
        action = self.available_actions[state][index]
        next_board, next_player = self.game.next_state(board, action, player)
        terminal_key = (next_board, action, player)
        if terminal_key not in self.terminal_state:
            self.terminal_state[terminal_key] = self.game.is_terminal_state(
                next_board, action, player
            )
        if self.terminal_state[terminal_key] is not None:
            value = self.game.compute_reward(self.terminal_state[terminal_key], player)
        else:
            value = self.search(next_board, next_player)
        self.__backup(state, index, value)
        return -value

    def __select(self, state):
        u = (
            self.args.c_puct
            * self.prior_probability[state]
            * numpy.sqrt(self.total_visit_count[state])
            / (1.0 + self.visit_count[state])
        )
        values = self.mean_action_value[state] + u
        return int(numpy.argmax(values))

    def __backup(self, state, index, value):
        self.mean_action_value[state][index] = (
            self.mean_action_value[state][index] * self.visit_count[state][index]
            + value
        ) / (self.visit_count[state][index] + 1.0)
        self.visit_count[state][index] += 1
        self.total_visit_count[state] += 1

    def __expand(self, board, player):
        state = self._state_key(board, player)
        canonical_board = self.game.get_canonical_form(board, player)
        proba, value = self.nnet.predict(canonical_board)
        actions = self.game.available_actions(board)
        self.available_actions[state] = actions
        if len(actions) == 0:
            self.prior_probability[state] = numpy.array([])
            self.total_visit_count[state] = 1
            self.mean_action_value[state] = numpy.zeros(0)
            self.visit_count[state] = numpy.zeros(0)
            return value
        self.prior_probability[state] = proba[actions] / numpy.sum(proba[actions])
        self.total_visit_count[state] = 1
        self.mean_action_value[state] = numpy.zeros(len(actions))
        self.visit_count[state] = numpy.zeros(len(actions))
        return value
