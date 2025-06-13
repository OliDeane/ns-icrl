from envs.two_agent_junction import TwoAgentJunction
import numpy as np


class SumoCombi2(TwoAgentJunction):
    def __init__(self, sim, args, discount=0.7):
        self.actions = [[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]]
        self.actions_str = ['stay', 'S', 'E', 'N', 'W']
        self.n_states = 2**8
        self.n_actions = len(self.actions)
        self.constraints = []

        self.goals = [(7, 4)]
        self.goal_str = ['east']
        self.n_goals = len(self.goals)

        self.p_transition = self._transition_prob_table()
        self.initial = self._get_initial_state_probs()
        self.terminal = self._get_terminal_states()

        self.valid_action = np.ones((self.n_states, self.n_actions))

    def compose_state(self, x, y, tls, r):
        s = x
        s += (y << 3)
        s += (tls << 6)
        s += (r << 7)

        return s

    def decompose_state(self, s):
        x = (s & 0x07)
        y = (s & 0x38) >> 3
        tls = (s & 0x40) >> 6
        r = (s & 0x80) >> 7

        return x, y, tls, r

    def on_road(self, s):
        x, y, _, _ = self.decompose_state(s)

        if y == 4:
            return True
        elif (x == 3) and (y < 5):
            return True
        elif (x == 6) and (y > 3):
            return True

        return False

    def traffic_light_violation(self, s, a):
        x, y, tls, r = self.decompose_state(s)

        # [0:'stay', 1:'S', 2:'E', 3:'N', 4:'W']

        if (x == 5) and (y == 4) and (tls == 1) and (a == 2):
            return True
        if (x == 6) and (y == 5) and (tls == 0) and (a == 1):
            return True

        return False

    def right_priority_violation(self, s, a):
        x, y, tls, r = self.decompose_state(s)

        if ((x == 2) and (y == 4) and (r == 1) and (a == 2)):
            return True
        return False

    def step(self, s, a, goal):
        s_ = np.random.choice(np.arange(self.n_states),
                              p=self.p_transition[s, :, a])

        done = False
        x, y, tls, r = self.decompose_state(s)
        x_, y_, tls_, r_ = self.decompose_state(s_)

        if not self.on_road(s_):
            c = 10
        elif self.traffic_light_violation(s, a):
            c = 10
        elif self.right_priority_violation(s, a):
            c = 10
        else:
            # l1 dist
            c = abs(x_-self.goals[goal][0]) + abs(y_-self.goals[goal][1])

        return s_, c, done

    def parse_answer_sets(self, args, file):
        s_a_map = np.ones((self.n_states, self.n_actions))

        with open(file) as f:
            for line in f:
                if line.startswith('row(1)'):
                    action_str = line.split('go(')[1].split(')')[0]

                    if action_str == 'north':
                        a = np.where(np.array(self.actions_str) == 'N')[0][0]
                    elif action_str == 'east':
                        a = np.where(np.array(self.actions_str) == 'E')[0][0]
                    elif action_str == 'south':
                        a = np.where(np.array(self.actions_str) == 'S')[0][0]
                    elif action_str == 'west':
                        a = np.where(np.array(self.actions_str) == 'W')[0][0]
                    elif action_str == 'zero':
                        a = np.where(np.array(self.actions_str)
                                     == 'stay')[0][0]
                    else:
                        raise Exception('invalid action string')

                    x = int(line.split('at(')[1][0])
                    y = int(line.split('at(')[1].split(',')[1].split(')')[0])
                    tls = int(line.split('tls')[1][0])
                    if 'carOnTheRight' in line:
                        r = 1
                    else:
                        r = 0

                    s = self.compose_state(x, y, tls, r)

                    s_a_map[s, a] = 0

        return s_a_map

    def load_constraints_from_hypothesis(self, args, gt=False, path=None):
        as_bg_path = f'ilasp/{args.env}/background_answer_sets.txt'

        if path != None:
            as_c_path = path
        else:
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

    def valid_state(self, s):
        x, y, tls, r = self.decompose_state(s)

        return (x > 0) and (y > 0) and (x < 8) and (y < 8)

    def _transition_prob(self, s_from, s_to, a):
        f_x, f_y, f_tls, f_r = self.decompose_state(s_from)
        t_x, t_y, t_tls, t_r = self.decompose_state(s_to)
        a = self.actions[a]

        if not self.valid_state(s_to):
            return 0.0

        if ((f_x + a[0]) == t_x) and ((f_y + a[1]) == t_y):
            # traffic light can change
            if (t_x == 2) and (t_y == 4):
                return 0.25
            elif t_r == 0.0:
                return 0.5
            else:
                return 0.0

            # right border, trying to go right
        if (f_x == 7) and (f_x == t_x) and (f_y == t_y) and (a[0] == 1) and (t_r == 0):
            return 0.5

        # left border, trying to go left
        if (f_x == 1) and (f_x == t_x) and (f_y == t_y) and (a[0] == -1) and (t_r == 0):
            return 0.5

        # upper border, trying to go up
        if (f_y == 7) and (f_x == t_x) and (f_y == t_y) and (a[1] == 1) and (t_r == 0):
            return 0.5

        # lower border, trying to go down
        if (f_y == 1) and (f_x == t_x) and (f_y == t_y) and (a[1] == -1) and (t_r == 0):
            return 0.5

        return 0.0

    def _get_initial_state_probs(self):
        initial = np.zeros((self.n_goals, self.n_states))

        for i in range(self.n_goals):
            for s in range(self.n_states):
                x, y, tls, r = self.decompose_state(s)

                if (x == 1) and (y == 4) and (r == 0):
                    initial[i, s] = 1.0/6
                elif (x == 3) and (y == 1) and (r == 0):
                    initial[i, s] = 1.0/6
                elif (x == 6) and (y == 7) and (r == 0):
                    initial[i, s] = 1.0/6

        return initial

    def _get_terminal_states(self):
        terminal_states = []

        for i in range(self.n_goals):
            terminal_states.append([])

            for s in range(self.n_states):
                x, y, tls, r = self.decompose_state(s)

                if(self.goals[i] == (x, y)):
                    terminal_states[i].append(s)

        return terminal_states
