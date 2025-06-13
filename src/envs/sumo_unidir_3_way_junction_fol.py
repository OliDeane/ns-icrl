from envs.two_agent_junction import TwoAgentJunction
import numpy as np
import json

# we should be able to approximate the environment with an MDP. How will
# we model the other agents?
# we do not assume we could interact with a simulator (or is this acceptable
# in this stage?).

# look for work which learns the dynamics of an environment from observations


class SumoUnidir3WayJunctionFol(TwoAgentJunction):
    def __init__(self, sim, args, discount=0.7):
        self.actions = sim['actions']
        self.actions_str = self.actions
        self.n_x = int(sim['x_max'] - sim['x_min']) + 1
        self.n_y = int(sim['y_max'] - sim['y_min']) + 1

        # road (0-3)
        # beforeJunction (0-1)
        # carLeft (0-1)
        # carInFront (0-1)
        # carRight (0-1)
        self.n_states = 2**6
        self.n_actions = len(self.actions)

        self.p_transition = self._transition_prob_table()

        self.terminal = self._get_terminal_states()
        self.reward = self._get_reward()
        self.initial = self._get_initial_state_probs()
        self.objective = self.reward

        self.discount = discount
        self.valid_action = np.ones((self.n_states, self.n_actions))
        self.feature_names = ["x", "y", "f", "r", "a"]

    def compose_state(self, road, beforeJunction, l, f, r):
        s = road
        s += (beforeJunction << 2)
        s += (l << 3)
        s += (f << 4)
        s += (r << 5)

        return s

    def decompose_state(self, s):
        road = (s & 0x03)
        beforeJunction = (s & 0x04) >> 2
        l = (s & 0x8) >> 3
        f = (s & 0x10) >> 4
        r = (s & 0x20) >> 5

        return road, beforeJunction, l, f, r

    def convert_constraint_to_array(self, c):
        return [c[0][0], c[0][1], c[0][2], c[0][3], c[1]]

    def _transition_prob(self, s_from, s_to, a):
        return 0.0

        # TODO
        # road, beforeJunction, l, f, r = self.decompose_state(s_from)
        # road_, beforeJunction_, l_, f_, r_ = self.decompose_state(s_to)
        # a = self.actions[a]
        #
        # # since this is a one way road, there can be no car on the left
        # if (l_ != 0):
        #     return 0.0
        #
        # if road == 1:
        #     if beforeJunction == 0:
        #         if road_ != 1:
        #             return 0.0
        #
        #         if beforeJunction_ == 0:
        #             if r_ != 0:
        #                 return 0.0
        #
        #             if a == 'straight':
        #                 return 0.5
        #             elif (a == 'stay') or (a == 'right'):
        #                 if f == 1:
        #                     return 0.5
        #                 elif (f == 0) and (f_ == 0):
        #                     return 1.0
        #                 else:
        #                     return 0.0
        #         else:
        #             if (a == 'stay') or (a == 'right'):
        #                 return 0.0
        #             elif a == 'straight':
        #                 return 0.25
        #     else:
        #         if a == 'straight':
        #             if (road_ != 0) or (r_ != 0):
        #                 return 0.0
        #
        # if beforeJunction == 0:
        #     # you cannot change roads when you are not at a junction
        #     if road != road_:
        #         return 0.0
        #
        #     if beforeJunction_ == 0:
        #         # When the next state is not at a junction, there can be no car on the right
        #         if r_ != 0:
        #             return 0.0
        #
        #         # when you go straight, there can be a car in front of you in the next state
        #         if a == 'straight':
        #             return 0.5
        #         elif (a == 'stay') or (a == 'right'):
        #             if f == 1:
        #                 return 0.5
        #             elif (f == 0) and (f_ == 0):
        #                 return 1.0
        #             else:
        #                 return 0.0
        #     else:
        #         # you can get at a junction where you are not driving
        #         if (a == 'stay') or (a == 'right'):
        #             return 0.0
        #
        #         # there can appair a car in front or at the right
        #         if a == 'straight':
        #             return 0.25
        # else:
        #     if (a == 'stay') and:
        #         return 0.25
        #
        #     # when you are at a junction

    def _get_initial_state_probs(self):
        initial = np.zeros(self.n_states)

        for s in range(self.n_states):
            x, y, f, r = self.decompose_state(s)

            if (x == 5) and (y == 0) and (f == 0) and (r == 0):
                initial[s] = 0.5
            elif (x == 0) and (y == 4) and (r == 0):
                initial[s] = 0.25

        return initial

    def _get_terminal_states(self):
        terminal_states = []

        for s in range(self.n_states):
            x, y, f, r = self.decompose_state(s)
            if (x == 7) and (y == 4):
                terminal_states.append(s)
            elif (x == 6) and (y == 4):
                terminal_states.append(s)
        return terminal_states
