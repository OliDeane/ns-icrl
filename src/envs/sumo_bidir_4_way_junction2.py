from PIL.Image import init
from envs.two_agent_junction import TwoAgentJunction
import numpy as np
import json
import os


class SumoBidir4WayJunction2(TwoAgentJunction):
    def __init__(self, sim, args, discount=0.7):
        self.actions = [[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]]
        self.actions_str = ['stay', 'S', 'E', 'N', 'W']
        self.n_x = args.gridsize #7#5  # 1, 2, 3, 4, 5
        self.n_y = args.gridsize #5
        self.constraints = []
        self.soft_constraints = []
        self.gridsize = args.gridsize

        self.goals = [(1, 3), (args.gridsize, 3), (3, 1), (3, args.gridsize)]
        self.goal_str = ['west', 'east', 'south', 'north']
        self.n_goals = len(self.goals)

        self.n_states = self.compute_num_states()#2**7
        self.n_actions = len(self.actions)
        self.p_transition = self._transition_prob_table()

        self.terminal = self._get_terminal_states()
        self.reward = self._get_reward()
        self.initial = self._get_initial_state_probs()
        self.objective = self.reward

        self.discount = discount
        self.valid_action = np.ones((self.n_states, self.n_actions))
        self.feature_names = ["x", "y", "tls", "a"]

        self.novel_candidate_elimination = args.novel_candidate_elimination

    def _compose_state(self, x, y, tls):
        s = x 
        s += (y << 3)
        s += (tls << 6)

        return s

    def _decompose_state(self, s):
        x = (s & 0x07)
        y = (s & 0x38) >> 3
        tls = (s & 0x40) >> 6

        return x, y, tls

    # def compose_state(self, x, y, tls):
    #     # Validate inputs
    #     if not (0 <= x < 10) or not (0 <= y < 10) or not (tls in [0, 1]):
    #         raise ValueError("Invalid values: x and y must be between 0 and 9, tls must be 0 or 1")

    #     s = x
    #     s += (y << 4)  # Shift y by 4 bits to the left
    #     s += (tls << 8)  # Shift tls by 8 bits to the left (4 bits for x + 4 bits for y)
    #     return s

    # def decompose_state(self, s):
    #     # Extract the values using masks and bit shifting
    #     x = (s & 0x0F)  # Mask with 0x0F (binary 00001111) to get the lower 4 bits for x
    #     y = (s & 0xF0) >> 4  # Mask with 0xF0 (binary 11110000) and shift right by 4 bits for y
    #     tls = (s & 0x100) >> 8  # Mask with 0x100 (binary 000100000000) and shift right by 8 bits for tls
    #     return x, y, tls


    def compose_state(self, x, y, tls):

        grid_size = self.gridsize

        # Validate inputs
        if not (0 <= x <= grid_size) or not (0 <= y <= grid_size) or not (tls in [0, 1]):
            print(x,y,tls)
            raise ValueError("Invalid values: x and y must be between 0 and grid_size, tls must be 0 or 1")

        # Calculate the number of bits needed to represent the grid size
        num_bits = (grid_size).bit_length()

        s = x
        s += (y << num_bits)  # Shift y by num_bits to the left
        s += (tls << (2 * num_bits))  # Shift tls by 2 * num_bits to the left
        return s
    
    def decompose_state(self, s):
        grid_size = self.gridsize

        # Calculate the number of bits needed to represent the grid size
        num_bits = (grid_size).bit_length()

        # Extract the values using masks and bit shifting
        x = (s & ((1 << num_bits) - 1))  # Mask with (1 << num_bits) - 1 to get the lower num_bits bits for x
        y = (s & (((1 << num_bits) - 1) << num_bits)) >> num_bits  # Mask with ((1 << num_bits) - 1) << num_bits and shift right by num_bits for y
        tls = (s & (1 << (2 * num_bits))) >> (2 * num_bits)  # Mask with 1 << (2 * num_bits) and shift right by 2 * num_bits for tls

        return x, y, tls




    def compute_num_states(self):
        print("Computing Number of States...")
        all_states = []
        for i in range(1,self.gridsize+1):
            for j in range(1,self.gridsize+1):
                for tls in [0,1]:
                    all_states.append(self.compose_state(i,j,tls))
        return max(all_states)


    def valid_state(self, s):
        x, y, tls = self.decompose_state(s)

        # return (x > 0) and (y > 0) and (x < 6) and (y < 6)
        return (x > 0) and (y > 0) and (x <= self.n_x) and (y <= self.n_y)


    def convert_constraint_to_array(self, c):
        return [c[0][0], c[0][1], c[0][2], c[1]]

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
                    s = self.compose_state(x, y, tls)

                    s_a_map[s, a] = 0

        return s_a_map

    def parse_constraints_to_map(self, constraints):
        s_a_map = np.zeros((self.n_states, self.n_actions))

        for constraint in constraints:
            x, y, tls = constraint[:3]
            s = self.compose_state(x, y, tls)
            a = constraint[-1]
            s_a_map[s, a] = 1

        return s_a_map

    def step(self, s, a, goal):
        print(self.p_transition.shape)
        print(s,a)
        print("----")
        print(self.p_transition[s, :, a])
        s_ = np.random.choice(np.arange(self.n_states),
                              p=self.p_transition[s, :, a])

        done = False
        x, y, tls = self.decompose_state(s)
        x_, y_, tls_ = self.decompose_state(s_)

        # [0:'stay', 1:'S', 2:'E', 3:'N', 4:'W']

        if (x_ != 3) and (y_ != 3):
            c = 10
        elif (x == 3) and (y == 2) and (tls == 0) and (a == 3):
            c = 10
        elif (x == 3) and (y == 4) and (tls == 0) and (a == 1):
            c = 10
        elif (x == 2) and (y == 3) and (tls == 1) and (a == 2):
            c = 10
        elif (x == 4) and (y == 3) and (tls == 1) and (a == 4):
            c = 10
        else:
            # l1 dist
            c = abs(x_-self.goals[goal][0]) + abs(y_-self.goals[goal][1])

        return s_, c, done

    def add_soft_constraint(self, args, s, a):
        if not self.valid_state(s):
            return False

        if not args.add_terminal_states:
            if s in np.array(self.terminal).flatten():
                return False

        c = (self.decompose_state(s), a)
        if c not in self.soft_constraints:
            self.soft_constraints.append(c)
            return True

        return False

    def load_raw_constraints(self, args, path):
        constraint_file = json.load(
            open(f'{path}/constraints.json', 'r'))['data']
        constraints = self.parse_constraints_to_map(constraint_file)

        for s in range(self.n_states):
            for a in range(self.n_actions):
                if constraints[s, a] == 1:
                    self.add_soft_constraint(args, s, a)

    def load_constraints_from_hypothesis(self, args, gt=False, soft_constraints=False):

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
                        if soft_constraints:
                            self.add_soft_constraint(args, s, a)
                        else:
                            self.add_constraint(args, s, a)

    def pothole_load_constraints_from_hypothesis(self, args, gt=False, soft_constraints=False):
        '''
        THIS HAS BEEN CHANGED TO HANDLE POTHOLES. REQUIRES CHANGING SO THAT ARGS CAN DETERMINE
        WHETHER POTHOLE FILES ARE SELECTED. Currently, it only works with the pothole example. And cannot be changes
        according to terminal arguments. 
        '''
        print(args.gridsize)
        as_bg_path = f'pothole_ilasp/{args.env}/background_answer_sets_{args.gridsize}grid.txt'
        if gt:
            as_c_path = f'pothole_ilasp/{args.env}/ground_truth_answer_sets_{args.gridsize}grid.txt'
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
                    if soft_constraints:
                        self.add_soft_constraint(args, s, a)
                    else:
                        self.add_constraint(args, s, a)

    def _transition_prob(self, s_from, s_to, a):
        f_x, f_y, f_tls = self.decompose_state(s_from)
        t_x, t_y, t_tls = self.decompose_state(s_to)
        a = self.actions[a]

        if not self.valid_state(s_to):
            return 0.0

        if ((f_x + a[0]) == t_x) and ((f_y + a[1]) == t_y):
            # traffic light can change
            return 0.5

        # right border, trying to go right
        if (f_x == self.n_x) and (f_x == t_x) and (f_y == t_y) and (a[0] == 1):
            return 0.5

        # left border, trying to go left
        if (f_x == 1) and (f_x == t_x) and (f_y == t_y) and (a[0] == -1):
            return 0.5

        # upper border, trying to go up
        if (f_y == self.n_y) and (f_x == t_x) and (f_y == t_y) and (a[1] == 1):
            return 0.5

        # lower border, trying to go down
        if (f_y == 1) and (f_x == t_x) and (f_y == t_y) and (a[1] == -1):
            return 0.5

        return 0.0

    def _get_initial_state_probs(self):
        print("Fetchuing initial state probs...")
        initial = np.zeros((self.n_goals, self.n_states))

        for i in range(self.n_goals):
            for s in range(self.n_states):
                x, y, tls = self.decompose_state(s)

                # all other goals are possible initial states
                for j in range(self.n_goals):
                    if (j != i) and ((x, y) == self.goals[j]):
                        initial[i, s] = 1.0/6

        return initial

    def _get_terminal_states(self):
        print("Fecthing Terminal States")
        terminal_states = []

        for i in range(self.n_goals):
            terminal_states.append([])

            for s in range(self.n_states):
                x, y, tls = self.decompose_state(s)

                if(self.goals[i] == (x, y)):
                    terminal_states[i].append(s)

        return terminal_states

    def _get_reward(self):
        print("Fetching Reward...")
        reward = np.zeros((self.n_goals, self.n_states, self.n_actions))

        for i in range(self.n_goals):
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    if s in self.terminal[i]:
                        reward[i, s, a] = 1.0
                    elif (self.decompose_state(s), a) in self.soft_constraints:
                        reward[i, s, a] = -0.5#-1000.0

        return reward
