from envs.two_agent_junction import TwoAgentJunction
import numpy as np

# we should be able to approximate the environment with an MDP. How will
# we model the other agents?
# we do not assume we could interact with a simulator (or is this acceptable
# in this stage?).

# look for work which learns the dynamics of an environment from observations


class SumoBidir3WayJunction(TwoAgentJunction):
    def __init__(self, sim, args, discount=0.7):
        self.actions = sim['actions']
        self.actions_str = ['stay', 'N', 'NW', 'W', 'E', 'SW', 'S']
        self.n_x = int(sim['x_max'] - sim['x_min']) + 1
        self.n_y = int(sim['y_max'] - sim['y_min']) + 1
        self.constraints = []

        self.n_states = 2**9
        self.n_actions = len(self.actions)
        self.p_transition = self._transition_prob_table()

        self.terminal = self._get_terminal_states(sim['goal'])
        self.reward = self._get_reward()
        self.initial = self._get_initial_state_probs()
        self.objective = self.reward

        self.discount = discount
        self.valid_action = np.ones((self.n_states, self.n_actions))
        self.feature_names = ["x", "y", "r", "goal", "goal_r", "a"]

    def compose_state(self, x, y, r, goal, goal_r):
        s = x
        s += (y << 3)
        s += (r << 5)
        # s += (goal << 6)
        # s += (goal_r << 8)

        if goal == 0:
            if goal_r == 0:
                goals = 0
            elif goal_r == 2:
                goals = 5
        elif goal == 1:
            if goal_r == 0:
                goals = 1
            elif goal_r == 1:
                goals = 3
            elif goal_r == 2:
                goals = 6
        elif goal == 2:
            if goal_r == 0:
                goals = 2
            elif goal_r == 1:
                goals = 4

        s += (goals << 6)

        return s

    def decompose_state(self, s):
        x = (s & 0x07)
        y = (s & 0x18) >> 3
        r = (s & 0x20) >> 5
        # goal = (s & 0xC0) >> 6
        # goal_r = (s & 0x300) >> 8
        goals = (s & 0x1C0) >> 6

        if goals == 0:
            goal = 0
            goal_r = 0
        elif goals == 1:
            goal = 1
            goal_r = 0
        elif goals == 2:
            goal = 2
            goal_r = 0
        elif goals == 3:
            goal = 1
            goal_r = 1
        elif goals == 4:
            goal = 2
            goal_r = 1
        elif goals == 5:
            goal = 0
            goal_r = 2
        elif goals == 6:
            goal = 1
            goal_r = 2

        return x, y, r, goal, goal_r

    def valid_state(self, s):
        x = (s & 0x07)
        y = (s & 0x18) >> 3
        r = (s & 0x20) >> 5
        goals = (s & 0x1C0) >> 6

        return (x < 6) and (goals < 7)

    def convert_constraint_to_array(self, c):
        return [c[0][0], c[0][1], c[0][2], c[0][3], c[0][4], c[1]]

    def _transition_prob(self, s_from, s_to, a):
        if (not self.valid_state(s_from)) or (not self.valid_state(s_to)):
            return 0.0

        f_x, f_y, f_r, f_goal, f_goal_r = self.decompose_state(s_from)
        t_x, t_y, t_r, t_goal, t_goal_r = self.decompose_state(s_to)
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
            # dep: down

            if (t_x == 3) and (t_y == 1):
                # at junction coming from down
                # possible goals: left(0), right(1)
                # car from the right (goal_r: left(0) or down(2))
                if t_r == 1:
                    if t_goal_r == 1:
                        return 0.0

                    # 3 possible goals
                    # 2 possible goals of agent to the right
                    return 1.0/3
                else:
                    if t_goal_r != 0:
                        return 0.0

                    # 3 possible goals
                    return 1.0/3

            elif (t_x == 1) and (t_y == 2):
                # add junction coming from the left
                # possible goals: down(2), right(1)
                # possibly car from the right (goal_r: left(0) or right(1))

                if t_r == 1:
                    if t_goal_r == 2:
                        return 0.0

                    # 3 possible goals
                    # 2 possible goals of agent to the right
                    return 1.0/6
                else:
                    if t_goal_r != 0:
                        return 0.0

                    # 3 possible goals
                    return 1.0/3

            else:
                # add junction coming from the right or not at junction
                if t_r != 0:
                    return 0.0
                if t_goal_r != 0:
                    return 0.0

                # 3 possible goals
                return 1.0/3

        # right border, trying to go right
        # TODO
        # this will not work for the diagonal actions
        if (f_x == (self.n_x - 1)) and (f_x == t_x) and (f_y == t_y) and (a[0] == 1) and (a[1] == 0):
            return 1.0

        # TODO
        # we should also the cases for the left and lower border

        # upper border, trying to go up
        if (f_y == (self.n_y - 1)) and (f_x == t_x) and (f_y == t_y) and (a[0] == 0) and (a[1] == 1):
            return 1.0

        return 0.0

    def _get_initial_state_probs(self):
        initial = np.zeros(self.n_states)

        for s in range(self.n_states):
            if self.valid_state(s):
                x, y, r, goal, goal_r = self.decompose_state(s)

                if (r == 0) and (goal_r == 0):
                    # check if goal is not the same as initial
                    # left
                    if (x == 0) and (y == 2):
                        if goal != 0:
                            initial[s] = 0.5
                    # right
                    elif (x == 5) and (y == 3):
                        if goal != 1:
                            initial[s] = 0.5
                    # down
                    elif (x == 3) and (y == 0):
                        if goal != 2:
                            initial[s] = 0.5

        return initial

    def _get_terminal_states(self, goals):
        terminal_states = []
        goals_x_y = []

        for state in goals:
            goals_x_y.append((state[0], state[1]))

        for s in range(self.n_states):
            if self.valid_state(s):
                x, y, r, goal, goal_r = self.decompose_state(s)

                # not correct: terminal_states depend on the goal
                # if the goal is to go west than only (0, 3) is a terminal state
                # since the reward is based on this
                # TODO
                if (x, y) in goals_x_y:
                    terminal_states.append(s)

        return terminal_states
