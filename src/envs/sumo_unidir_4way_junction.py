from envs.two_agent_junction import TwoAgentJunction
import numpy as np
import json
import os


class SumoUnidir4WayJunction(TwoAgentJunction):
    def __init__(self, sim, args, discount=0.7):
        self.actions = [[0, 0], [1, 0], [0, 1], [0, -1], [-1, 0]]
        self.actions_str = ['stay', 'E', 'N', 'S', 'W']
        self.n_x = 5
        self.n_y = 5
        self.constraints = []

        self.goals = [(5, 3)]
        self.goal_str = ['east']
        self.n_goals = len(self.goals)

        self.n_states = 2**7
        self.n_actions = len(self.actions)
        self.p_transition = self._transition_prob_table()

        self.terminal = self._get_terminal_states()
        self.reward = self._get_reward()
        self.initial = self._get_initial_state_probs()
        self.objective = self.reward

        self.discount = discount
        self.valid_action = np.ones((self.n_states, self.n_actions))
        self.feature_names = ["x", "y", "r", "a"]

    def compose_state(self, x, y, r):
        s = x
        s += (int(y) << 3)
        s += (int(r) << 6)

        return s

    def decompose_state(self, s):
        x = (int(s) & 0x07)
        y = (int(s) & 0x38) >> 3
        r = (int(s) & 0x40) >> 6

        return x, y, r

    def convert_constraint_to_array(self, c):
        return [c[0][0], c[0][1], c[0][2], c[1]]

    def valid_state(self, s):
        x, y, r = self.decompose_state(s)

        if not((x > 0) and (y > 0) and (x < 6) and (y < 6)):
            return False
        if (r == 1) and not (((x == 2) and (y == 3)) or ((x == 3) and (y == 4))):
            return False

        return True

    def parse_answer_sets(self, args, file):
        s_a_map = np.ones((self.n_states, self.n_actions))

        with open(file) as f:
            for line in f:
                if line.startswith('row(1)'):
                    if 'go(zero)' in line:
                        a = np.where(np.array(self.actions_str) == 'stay')
                    elif 'go(east)' in line:
                        a = np.where(np.array(self.actions_str) == 'E')
                    elif 'go(north)' in line:
                        a = np.where(np.array(self.actions_str) == 'N')
                    elif 'go(south)' in line:
                        a = np.where(np.array(self.actions_str) == 'S')
                    elif 'go(west)' in line:
                        a = np.where(np.array(self.actions_str) == 'W')
                    else:
                        raise Exception('no action found in line')

                    x = int(line.split('at(')[1][0])
                    y = int(line.split('at(')[1].split(',')[1].split(')')[0])
                    r = 0
                    if 'carOnTheRight' in line:
                        r = 1

                    s = self.compose_state(x, y, r)
                    s_a_map[s, a] = 0

        return s_a_map

    def parse_constraints_to_map(self, constraints):
        s_a_map = np.zeros((self.n_states, self.n_actions))

        for constraint in constraints:
            x, y, r = constraint[:3]
            s = self.compose_state(x, y, r)
            a = constraint[-1]
            s_a_map[s, a] = 1

        return s_a_map

    def load_constraints_from_hypothesis(self, args, gt=False):
        as_bg_path = f'ilasp/{args.env}/background_answer_sets.txt'

        if gt:
            as_c_path = f'ilasp/{args.env}/ground_truth_answer_sets.txt'
        else:
            as_c_path = f'results/{args.env}/run{args.run}_c{args.eta-1}_o{args.num_observations}/answer_sets.txt'

        as_bg = self.parse_answer_sets(args, as_bg_path)
        as_c = self.parse_answer_sets(args, as_c_path)

        if np.sum(as_c) < np.sum(as_bg):
            raise Exception('invalid answer sets')

        diff = as_bg - as_c
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if diff[s, a] == -1:
                    self.add_constraint(args, s, a)

    def load_raw_constraints(self, args, path):
        constraint_file = json.load(
            open(f'{path}/constraints.json', 'r'))['data']
        constraints = self.parse_constraints_to_map(constraint_file)

        for s in range(self.n_states):
            for a in range(self.n_actions):
                if constraints[s, a] == 1:
                    self.add_constraint(args, s, a)

    def step(self, s, a, goal):
        s_ = np.random.choice(np.arange(self.n_states),
                              p=self.p_transition[s, :, a])

        done = False
        x, y, r = self.decompose_state(s)
        x_, y_, r_ = self.decompose_state(s_)

        if (x_ != 3) and (y_ != 3):
            c = 10
        elif (x == 3) and (y == 4) and (r == 1) and (a != 0):
            c = 10
        elif (x == 2) and (y == 3) and (r == 1) and (a != 0):
            c = 10
        else:
            # l1 dist
            c = abs(x_-self.goals[goal][0]) + abs(y_-self.goals[goal][1])

        return s_, c, done

    def _transition_prob(self, s_from, s_to, a):
        f_x, f_y, f_r = self.decompose_state(s_from)
        t_x, t_y, t_r = self.decompose_state(s_to)
        a = self.actions[a]

        if not self.valid_state(s_to):
            return 0.0

        if ((f_x + a[0]) == t_x) and ((f_y + a[1]) == t_y):
            if ((t_x == 2) and (t_y == 3)) or ((t_x == 3) and (t_y == 4)):
                return 0.5
            elif (t_r == 0):
                return 1.0
            else:
                return 0.0

        # right border, trying to go right
        if (f_x == 5) and (f_x == t_x) and (f_y == t_y) and (a[0] == 1) and (a[1] == 0) and (t_r == 0):
            return 1.0

        # left border, trying to go left
        if (f_x == 1) and (f_x == t_x) and (f_y == t_y) and (a[0] == -1) and (t_r == 0):
            return 1.0

        # upper border, trying to go up
        if (f_y == 5) and (f_x == t_x) and (f_y == t_y) and (a[0] == 0) and (a[1] == 1) and (t_r == 0):
            return 1.0

        # lower border, trying to go down
        if (f_y == 1) and (f_x == t_x) and (f_y == t_y) and (a[1] == -1) and (t_r == 0):
            return 1.0

        return 0.0

    def _get_initial_state_probs(self):
        initial = np.zeros((self.n_goals, self.n_states))

        for g in range(self.n_goals):
            for s in range(self.n_states):
                x, y, r = self.decompose_state(s)

                if (x == 3) and (y == 1) and (r == 0):
                    initial[g, s] = 1.0/3
                elif (x == 1) and (y == 3) and (r == 0):
                    initial[g, s] = 1.0/3
                elif (x == 3) and (y == 5) and (r == 0):
                    initial[g, s] = 1.0/3

        return initial

    def _get_terminal_states(self):
        terminal_states = []

        for i in range(self.n_goals):
            terminal_states.append([])

            for s in range(self.n_states):
                x, y, r = self.decompose_state(s)

                if(self.goals[i] == (x, y)):
                    terminal_states[i].append(s)

        return terminal_states

    def _get_reward(self):
        reward = np.zeros((self.n_goals, self.n_states, self.n_actions))

        for i in range(self.n_goals):
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    if s in self.terminal[i]:
                        reward[i, s, a] = 1.0

        return reward
