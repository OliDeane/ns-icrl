from itertools import product
from envs.simple_junction import SimpleJunction
import numpy as np


class TwoAgentJunction(SimpleJunction):
    def __init__(self, constrained=False, discount=0.7):
        self.actions = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
        self.actions_str = ['stay', 'right', 'up', 'left', 'down']

        self.n_x = 5
        self.n_y = 5

        # In each position the other agent can be left, in front or
        # right of the current agent or the other agent is not visible
        self.n_states = (self.n_x * self.n_y) * 4
        self.n_actions = len(self.actions)

        self.p_transition = self._transition_prob_table()

        self.terminal = self._get_terminal_states()
        self.reward = self._get_reward()
        self.initial = self._get_initial_state_probs()
        self.objective = self.reward

        self.discount = discount
        self.valid_action = np.ones((self.n_states, self.n_actions))
        self.feature_names = ["x", "y", "l", "f", "r", "a"]

        if constrained:
            self.augment_with_constraints()

    def augment_with_constraints(self):
        # only drive on the road
        for s in range(self.n_states):
            x, y, l, f, r = self.decompose_state(s)

            if (x < 3) and (y < 3):
                self.add_state_constraint(s)
            elif (x == 4) and (y != 3):
                self.add_state_constraint(s)
            elif (y == 4):
                self.add_state_constraint(s)

        # priority of the right
        self.add_constraint(self.compose_state(x=2, y=3, l=0, f=1, r=0), 1)
        self.add_constraint(self.compose_state(x=2, y=3, l=0, f=0, r=1), 1)

    def compose_state(self, x, y, l, f, r):
        if l + f + r > 1:
            raise Exception(
                "there can only be one second agent, l + f + r <= 1")
        s = 0
        s += (y * self.n_x) + x
        s += (l * 1) * self.n_x * self.n_y
        s += (f * 2) * self.n_x * self.n_y
        s += (r * 3) * self.n_x * self.n_y

        return s

    def decompose_state(self, s):
        x = s % self.n_x
        y = (s % (self.n_x * self.n_y)) // self.n_x
        l = 0
        f = 0
        r = 0

        if s > 74:
            r = 1
        elif s > 49:
            f = 1
        elif s > 24:
            l = 1

        return x, y, l, f, r

    def convert_constraint_to_array(self, c):
        return [c[0][0], c[0][1], c[0][2], c[0][3], c[0][4], c[1]]

    def _transition_prob(self, s_from, s_to, a):
        f_x, f_y, f_l, f_f, f_r = self.decompose_state(s_from)
        t_x, t_y, t_l, t_f, t_r = self.decompose_state(s_to)
        a = self.actions[a]

        if ((f_x + a[0]) == t_x) and ((f_y + a[1]) == t_y):
            return 0.25

        # left border, trying to go left
        if (f_x == 0) and (f_x == t_x) and (f_y == t_y) and (a[0] == -1) and (a[1] == 0):
            return 0.25

        # right border, trying to go right
        if (f_x == (self.n_x - 1)) and (f_x == t_x) and (f_y == t_y) and (a[0] == 1) and (a[1] == 0):
            return 0.25

        # lower border, trying to go down
        if (f_y == 0) and (f_x == t_x) and (f_y == t_y) and (a[0] == 0) and (a[1] == -1):
            return 0.25

        # upper border, trying to go up
        if (f_y == (self.n_y - 1)) and (f_x == t_x) and (f_y == t_y) and (a[0] == 0) and (a[1] == 1):
            return 0.25

        return 0.0

    def _get_terminal_states(self):
        terminal_states = []

        for s in range(self.n_states):
            x, y, l, f, r = self.decompose_state(s)
            if (x == 4) and (y == 3):
                terminal_states.append(s)

        return terminal_states

    def _get_reward(self):
        reward = np.zeros((self.n_goals, self.n_states, self.n_actions))

        for g in range(self.n_goals):
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    if s in self.terminal[g]:
                        reward[g, s, a] = 1.0

        return reward

    def _get_initial_state_probs(self):
        initial = np.zeros((self.n_goals, self.n_states))

        for g in range(self.n_goals):
            for s in range(self.n_states):
                x, y, l, f, r = self.decompose_state(s)

                if (x == 3) and (y == 0):
                    initial[g, s] = 0.125
                elif (x == 0) and (y == 3):
                    initial[g, s] = 0.125

        return initial
