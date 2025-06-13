from PIL.Image import init
from envs.two_agent_junction import TwoAgentJunction
import numpy as np


class SumoBidir4WayJunction(TwoAgentJunction):
    def __init__(self, sim, args, discount=0.7):
        self.actions = sim['actions']
        self.actions_str = ['stay', 'S', 'E', 'W', 'N', 'NW', 'SW', 'NE', 'SE']
        self.n_x = 6
        self.n_y = 6
        self.constraints = []

        self.g_l = 0
        self.g_r = 1
        self.g_d = 2
        self.g_u = 3

        self.n_states = 2**9
        self.n_actions = len(self.actions)
        self.p_transition = self._transition_prob_table()

        self.terminal = self._get_terminal_states(sim['goal'])
        self.reward = self._get_reward()
        self.initial = self._get_initial_state_probs()
        self.objective = self.reward

        self.discount = discount
        self.valid_action = np.ones((self.n_states, self.n_actions))
        self.feature_names = ["x", "y", "tls", "goal", "a"]

    def compose_state(self, x, y, tls, goal):
        s = x
        s += (y << 3)
        s += (tls << 6)
        s += (goal << 7)

        return s

    def decompose_state(self, s):
        x = (s & 0x07)
        y = (s & 0x38) >> 3
        tls = (s & 0x40) >> 6
        goal = (s & 0x180) >> 7

        return x, y, tls, goal

    def valid_state(self, s):
        x, y, tls, goal = self.decompose_state(s)

        return (x < 6) and (y < 6)

    def convert_constraint_to_array(self, c):
        return [c[0][0], c[0][1], c[0][2], c[0][3], c[1]]

    def _transition_prob(self, s_from, s_to, a):
        f_x, f_y, f_tls, f_goal = self.decompose_state(s_from)
        t_x, t_y, t_tls, t_goal = self.decompose_state(s_to)
        a = self.actions[a]

        # goal cannot change
        if f_goal != t_goal:
            return 0.0

        # x outside of state space
        if t_x > (self.n_x - 1):
            return 0.0

        # y outside of state space
        if t_y > (self.n_y - 1):
            return 0.0

        if ((f_x + a[0]) == t_x) and ((f_y + a[1]) == t_y):
            # traffic light can change
            return 0.5

        # right border, trying to go right
        if (f_x == (self.n_x - 1)) and (f_x == t_x) and (f_y == t_y) and (a[0] == 1):
            return 1.0

        # left border, trying to go left
        if (f_x == 0) and (f_x == t_x) and (f_y == t_y) and (a[0] == -1):
            return 1.0

        # upper border, trying to go up
        if (f_y == (self.n_y - 1)) and (f_x == t_x) and (f_y == t_y) and (a[1] == 1):
            return 1.0

        # lower border, trying to go down
        if (f_y == 0) and (f_x == t_x) and (f_y == t_y) and (a[1] == -1):
            return 1.0

        return 0.0

    def _get_initial_state_probs(self):
        initial = np.zeros(self.n_states)

        i_l = (0, 2)
        i_r = (5, 3)
        i_d = (3, 0)
        i_u = (2, 5)

        for s in range(self.n_states):
            x, y, tls, goal = self.decompose_state(s)

            if goal == self.g_l:
                if ((x, y) == i_r) or ((x, y) == i_d) or ((x, y) == i_u):
                    initial[s] = 1.0/3
            elif goal == self.g_r:
                if ((x, y) == i_l) or ((x, y) == i_d) or ((x, y) == i_u):
                    initial[s] = 1.0/3
            elif goal == self.g_d:
                if ((x, y) == i_l) or ((x, y) == i_r) or ((x, y) == i_u):
                    initial[s] = 1.0/3
            elif goal == self.g_u:
                if ((x, y) == i_l) or ((x, y) == i_d) or ((x, y) == i_r):
                    initial[s] = 1.0/3

        return initial

    def _get_terminal_states(self, goals):
        terminal_states = []

        g_l = (0, 3)
        g_r = (5, 2)
        g_d = (2, 0)
        g_u = (3, 5)

        for s in range(self.n_states):
            x, y, tls, goal = self.decompose_state(s)

            if ((goal == self.g_l) and ((x, y) == g_l)) or \
                ((goal == self.g_r) and ((x, y) == g_r)) or \
                ((goal == self.g_d) and ((x, y) == g_d)) or \
                    ((goal == self.g_u) and ((x, y) == g_u)):
                terminal_states.append(s)

        return terminal_states
