from envs.two_agent_junction import TwoAgentJunction
import numpy as np
import json

# we should be able to approximate the environment with an MDP. How will
# we model the other agents?
# we do not assume we could interact with a simulator (or is this acceptable
# in this stage?).

# look for work which learns the dynamics of an environment from observations


class SumoUnidir3WayJunction(TwoAgentJunction):
    def __init__(self, sim, args, discount=0.7):
        self.actions = [[0, 0], [1, 0], [0, 1]]
        self.actions_str = ['stay', 'right', 'up']
        self.n_x = 7  # int(sim['x_max'] - sim['x_min']) + 1
        self.n_y = 5  # int(sim['y_max'] - sim['y_min']) + 1

        # In each position the other agent can be left, in front or
        # right of the current agent or the other agent is not visible
        self.n_states = 2**7
        self.n_actions = len(self.actions)
        self.n_goals = 1

        self.p_transition = self._transition_prob_table()

        self.terminal = self._get_terminal_states()
        self.reward = self._get_reward()
        self.initial = self._get_initial_state_probs()
        self.objective = self.reward

        self.discount = discount
        self.valid_action = np.ones((self.n_states, self.n_actions))
        self.feature_names = ["x", "y", "r", "a"]
        self.constraints = []

    def compose_state(self, x, y, r):
        s = x
        s += (y << 3)
        s += (r << 6)

        return s

    def decompose_state(self, s):
        x = (s & 0x07)
        y = (s & 0x38) >> 3
        r = (s & 0x40) >> 6

        return x, y, r

    def convert_constraint_to_array(self, c):
        return [c[0][0], c[0][1], c[0][2], c[1]]

    def valid_state(self, s):
        x, y, r = self.decompose_state(s)

        if x > 6:
            return False
        if y > 4:
            return False
        if (r == 1) and not ((x == 2) and (y == 4)):
            return False

        return True

    def parse_answer_sets(self, args, file):
        s_a_map = np.ones((self.n_states, self.n_actions))

        with open(file) as f:
            for line in f:
                if line.startswith('row(0)'):
                    if 'stop' in line:
                        a = np.where(np.array(self.actions_str) == 'stay')
                    elif 'driveEast' in line:
                        a = np.where(np.array(self.actions_str) == 'right')
                    elif 'driveNorth' in line:
                        a = np.where(np.array(self.actions_str) == 'up')
                    else:
                        raise Exception('no action found in line')

                    x = int(line.split('at(')[1][0])
                    y = int(line.split('at(')[1].split(',')[1].split(')')[0])
                    r = 0
                    if 'carRight' in line:
                        r = 1

                    s = self.compose_state(x, y, r)
                    s_a_map[s, a] = 0

        return s_a_map

    def _transition_prob(self, s_from, s_to, a):
        f_x, f_y, f_r = self.decompose_state(s_from)
        t_x, t_y, t_r = self.decompose_state(s_to)
        a = self.actions[a]

        if not self.valid_state(s_to):
            return 0.0

        if t_x > (self.n_x - 1):
            return 0.0

        if t_y > (self.n_y - 1):
            return 0.0

        if ((f_x + a[0]) == t_x) and ((f_y + a[1]) == t_y):
            if (t_x == 2) and (t_y == 4):
                return 0.5
            elif (t_r == 0):
                return 1.0
            else:
                return 0.0

        # right border, trying to go right
        if (f_x == (self.n_x - 1)) and (f_x == t_x) and (f_y == t_y) and (a[0] == 1) and (a[1] == 0):
            return 1.0

        # upper border, trying to go up
        if (f_y == (self.n_y - 1)) and (f_x == t_x) and (f_y == t_y) and (a[0] == 0) and (a[1] == 1):
            return 1.0

        return 0.0

    def _get_initial_state_probs(self):
        initial = np.zeros((self.n_goals, self.n_states))

        for g in range(self.n_goals):
            for s in range(self.n_states):
                x, y, r = self.decompose_state(s)

                if (x == 3) and (y == 0) and (r == 0):
                    initial[g, s] = 0.5
                elif (x == 0) and (y == 4) and (r == 0):
                    initial[g, s] = 0.5

        return initial

    def _get_terminal_states(self):
        terminal_states = []

        for s in range(self.n_states):
            x, y, r = self.decompose_state(s)
            if (y == 4) and (x == 6):
                terminal_states.append(s)
        return [terminal_states]
