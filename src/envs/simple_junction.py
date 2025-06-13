from envs.constrained_mdp import ConstrainedMdp
import numpy as np


class SimpleJunction(ConstrainedMdp):
    def __init__(self, size, constrained=False, goal='left', discount=0.7):
        # (x, y)
        self.actions = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
        self.actions_str = ['stay', 'right', 'up', 'left', 'down']

        self.n_x = size
        self.n_y = size

        # only one traffic light,
        # agent always comes from the same direction
        self.n_tl = 1

        self.n_states = (size**2)*(2**self.n_tl)
        self.n_actions = len(self.actions)

        self.p_transition = self._transition_prob_table()

        self.goal = goal
        self.goal_actions = {
            'left': 3,
            'right': 1,
            'top': 2,
            'bottom': 1
        }

        # do not change the order of these function calls
        if type(self) == SimpleJunction:
            self.terminal = self._get_terminal_states(goal)
            self.reward = self._get_reward()
            self.initial = self._get_initial_state_probs()
            self.objective = self.reward

        self.feature_names = ["x", "y", "tl", "a"]
        self.discount = discount
        self.valid_action = np.ones((self.n_states, self.n_actions))
        self.constraints = []

        # augment constraints on MDP
        if constrained:
            self.augment_with_constraints()

    def augment_with_constraints(self):
        # stay on the road
        for s in range(self.n_states):
            for a in range(self.n_actions):
                x, y, _ = self.decompose_state(s)

                # check if x,y is on the road
                if not ((x == (self.n_x // 2)) or (y == (self.n_y // 2))):
                    self.add_constraint(s, a)
                else:
                    # check if the chosen action does not drive the car of the road
                    for s_ in range(self.n_states):
                        if self.p_transition[s, s_, a] > 0:
                            x_, y_, _ = self.decompose_state(s_)

                            if not ((x_ == (self.n_x // 2)) or (y_ == (self.n_y // 2))):
                                self.add_constraint(s, a)

        # stop before traffic light
        self.add_constraint(self.compose_state(
            self.n_x//2, (self.n_y//2) - 1, 1), 2)

    def convert_constraint_to_array(self, c):
        """
            convert a constraint which consists of multiple tuples and arrays to one clean array
        """

        return [c[0][0], c[0][1], c[0][2], c[1]]

    def decompose_state(self, s):
        """
            returns the x- and y-coordinate and the state of the traffic light given the state index

            feature vector: | X | tl_n | ... | tl |
        """

        x = s % self.n_x
        y = (s % (self.n_x * self.n_y)) // self.n_x
        tl = s // (self.n_x * self.n_y)

        return x, y, tl

    def compose_state(self, x, y, tl):
        """
            Given an x- and y-coordinate and the state of the traffic light, return the state index

            feature vector: | X | tl_n | ... | tl |
        """
        s = 0
        s += (y * self.n_x) + x
        s += tl * self.n_x * self.n_y  # this works because there is only one traffic light

        return s

    def decompose_state_action_pair(self, s_a):
        return s_a // self.n_actions, s_a % self.n_actions

    def compose_state_action_pair(self, s, a):
        return (s * self.n_actions) + a

    def _transition_prob(self, s_from, s_to, a):
        f_x, f_y, f_tl = self.decompose_state(s_from)
        t_x, t_y, t_tl = self.decompose_state(s_to)
        a = self.actions[a]

        if ((f_x + a[0]) == t_x) and ((f_y + a[1]) == t_y):
            return 1.0 / 2**self.n_tl

        # left border, trying to go left
        if (f_x == 0) and (f_x == t_x) and (f_y == t_y) and (a[0] == -1) and (a[1] == 0):
            return 1.0 / 2**self.n_tl

        # right border, trying to go right
        if (f_x == (self.n_x - 1)) and (f_x == t_x) and (f_y == t_y) and (a[0] == 1) and (a[1] == 0):
            return 1.0 / 2**self.n_tl

        # lower border, trying to go down
        if (f_y == 0) and (f_x == t_x) and (f_y == t_y) and (a[0] == 0) and (a[1] == -1):
            return 1.0 / 2**self.n_tl

        # upper border, trying to go up
        if (f_y == (self.n_y - 1)) and (f_x == t_x) and (f_y == t_y) and (a[0] == 0) and (a[1] == 1):
            return 1.0 / 2**self.n_tl

        return 0.0

    def _get_terminal_states(self, goal):
        terminal_states = []

        for i in range(2**self.n_tl):
            if goal == 'left':
                terminal_states.append(self.compose_state(
                    x=0, y=self.n_y // 2, tl=i))
            elif goal == 'right':
                terminal_states.append(self.compose_state(
                    x=self.n_x - 1, y=self.n_y // 2, tl=i))
            elif goal == 'up':
                terminal_states.append(self.compose_state(
                    x=self.n_x // 2, y=self.n_y - 1, tl=i))
            else:
                raise Exception('goal: {} not specified'.format(goal))

        return terminal_states

    def _get_reward(self):
        reward = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            for a in range(self.n_actions):
                # and (a == self.goal_actions[self.goal]):
                if s in self.terminal:
                    reward[s][a] = 1.0

        return reward

    def _get_initial_state_probs(self):
        initial = np.zeros(self.n_states)

        for i in range(2**self.n_tl):
            initial[self.compose_state(x=self.n_x // 2, y=0, tl=i)] = 0.5

        return initial
