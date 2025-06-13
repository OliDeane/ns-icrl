import numpy as np
from itertools import product


class SimpleLane:
    """
    Simple one way lane with a traffic light at x=2. 
    Each position along the x-axis corresponds with two states, 
    a state where the traffic light is on and a state where the traffic light is off.

    size: length of the lane environment
    num_tl: number of traffic lights
    """

    def __init__(self, size, num_tl):
        self.actions = [0, 1]

        # possible x and tl states
        self.n_x = size
        self.n_tl = num_tl

        # states are duplicated because of the traffic light
        self.n_states = size*(2**self.n_tl)
        self.n_actions = len(self.actions)

        # transition probability function
        self.p_transition = self._transition_prob_table()

        self.terminal = self._get_terminal_states()
        self.reward = self._get_reward()
        self.initial = self._get_initial_state_probs()
        self.objective = self.reward

        self.feature_names = ["x", "tl0", "tl1", "a"]

    def convert_constraint_to_array(self, c):
        """
            convert a constraint which consists of multiple tuples and arrays to one clean array
        """

        return [c[0][0], int(c[0][1][0]), int(c[0][1][1]), c[1]]

    def decompose_state(self, s):
        """
            returns the x-coordinate and the state of the traffic lights

            feature vector: | X | tl_n | ... | tl_1 | tl_0 |
        """

        x = s // (2**self.n_tl)
        s -= x * (2**self.n_tl)

        tls = self.get_tl_vector(s)

        return x, tls

    def get_tl_vector(self, tl):
        tl_v = np.zeros(self.n_tl)

        for i in range(self.n_tl - 1, 0, -1):
            tl_v[self.n_tl - 1 - i] = tl // (2**i)
            tl -= tl_v[self.n_tl - 1 - i] * (2**i)

        tl_v[-1] = tl % 2

        return tl_v

    def compose_state(self, x, tls):
        """
            Given an x-coordinate and the state of the traffic light, return the state index

            feature vector: | X | tl_n | ... | tl_1 | tl_0 |
        """
        s = 0
        s += x * (2**self.n_tl)
        s += self.get_tl_repr(tls)

        return s

    def get_tl_repr(self, tls):
        s_tl = 0

        for i in range(self.n_tl - 1, -1, -1):
            s_tl += tls[self.n_tl - 1 - i] * (2**i)

        return int(s_tl)

    def decompose_state_action_pair(self, s_a):
        return s_a // self.n_actions, s_a % self.n_actions

    def compose_state_action_pair(self, s, a):
        return (s * self.n_actions) + a

    def _transition_prob_table(self):
        """
        Builds the internal probability transition table.

        Returns:
            The probability transition table of the form

                [state_from, state_to, action]

            containing all transition probabilities. The individual
            transition probabilities are defined by `self._transition_prob'.
        """
        table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))

        s1, s2, a = range(self.n_states), range(
            self.n_states), range(self.n_actions)
        for s_from, s_to, a in product(s1, s2, a):
            table[s_from, s_to, a] = self._transition_prob(s_from, s_to, a)

        return table

    def _transition_prob(self, s_from, s_to, a):
        """
        Compute the transition probability for a single transition.

        Args:
            s_from: The state in which the transition originates.
            s_to: The target-state of the transition.
            a: The action via which the target state should be reached.

        Returns:
            The transition probability from `s_from` to `s_to` when taking
            action `a`.
        """
        f_x, f_tl = self.decompose_state(s_from)
        t_x, t_tl = self.decompose_state(s_to)
        a = self.actions[a]

        if f_x + a == t_x:
            return 1.0 / 2**self.n_tl
        elif (f_x == t_x) and (f_x == (self.n_x - 1)):
            return 1.0 / 2**self.n_tl

        return 0.0

    def _get_terminal_states(self):
        terminal_states = []

        for i in range(2**self.n_tl):
            terminal_states.append(self.n_states-i-1)

        return terminal_states

    def _get_reward(self):
        reward = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            for a in range(self.n_actions):
                x, _ = self.decompose_state(s)

                if x == (self.n_x-1):
                    reward[s][a] = 1.0

        return reward

    def _get_initial_state_probs(self):
        initial = np.zeros(self.n_states)

        for i in range(2**self.n_tl):
            initial[i] = 1.0 / 2**self.n_tl

        return initial


class SimpleLaneConstrained(SimpleLane):
    def __init__(self, size, num_tl):
        SimpleLane.__init__(self, size, num_tl)

        self.valid_action = np.ones((self.n_states, self.n_actions))

    def get_valid_actions(self, state):
        """
        Get the possible actions given a state
        """

        valid_actions = []
        for i in range(self.n_actions):
            if self.valid_action[state, i] == 1:
                valid_actions.append(i)

        return np.array(valid_actions)

    def add_constraint(self, s, a):
        self.valid_action[s, a] = 0


class SimpleLaneConstrainedWithCostFunction(SimpleLane):
    def __init__(self, size):
        SimpleLane.__init__(self, size)

        # lagrange multiplier
        self.l = 10
        # budget alpha
        self.alpha = 0.1

        # cost function
        self.cost = np.zeros((self.n_states, self.n_actions))

        # objective which combines the reward and the cost function
        self._update_objective()

    def add_constraint(self, s, a):
        self.cost[s, a] = 1.0
        self._update_objective()

    def _update_objective(self):
        self.objective = self.reward - (self.l * (self.cost - self.alpha))
