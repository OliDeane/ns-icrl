from copy import deepcopy
from envs.constrained_mdp import ConstrainedMdp
import numpy as np
import itertools
import os


class Hanoi(ConstrainedMdp):
    def __init__(self, constrained=True):
        self.n_disks = 3
        self.n_pegs = 3
        self.states = list(itertools.product([(x, y) for x in range(self.n_pegs)
                                              for y in range(self.n_disks)], repeat=self.n_disks))
        self.states = np.array(self.states)
        self.n_states = len(self.states)
        self.n_goals = 1
        self.constrained = constrained
        self.constraints = []

        # an action is a tuple of two elements where the first element
        # corresponds with the disk index and the second element is
        # the new position of that disk
        self.actions = [(disk, peg) for disk in range(self.n_disks)
                        for peg in range(self.n_pegs)]
        self.actions_str = [
            'disk {} -> peg {}'.format(action[0], action[1]) for action in self.actions]

        self.n_actions = len(self.actions)
        self.initial_state_ind = self.get_state_ind(self._get_initial_state())
        self.valid_action = np.ones((self.n_states, self.n_actions))
        self.discount = 0.7

        self.p_transition = self._get_p_transition_table_from_disk('hanoi')
        self.objective = self._get_reward()
        self.initial = self._get_initial_state_probs()
        self.terminal = np.array(
            [[self.get_state_ind(self._get_goal_state())]])
        self.feature_names = ['peg_d0', 'height_d0',
                              'peg_d1', 'height_d1', 'peg_d1', 'height_d1', 'a']

    def decompose_state(self, state):
        # TODO remove self.get_state_from_ind and replace with decompose state
        return tuple([tuple(elem) for elem in self.get_state_from_ind(state)])

    def get_state_ind(self, state):
        return np.argwhere((self.states == state).all(axis=(1, 2)))[0][0]

    def get_state_from_ind(self, state_ind):
        return self.states[state_ind]

    def decompose_state_action_pair(self, s_a):
        return s_a // self.n_actions, s_a % self.n_actions

    def convert_constraint_to_array(self, constraint):
        s, a = constraint
        arr = [e for elem in s for e in elem]
        arr.append(a)

        return arr

    def parse_answer_sets(self, args, file):
        s_a_map = np.ones((self.n_states, self.n_actions))

        with open(file) as f:
            for line in f:
                if line.startswith('y(0)'):
                    state = np.zeros((3, 2))

                    for cords in line.split('at(')[1:]:
                        d, x, y = [int(elem[0])
                                   for elem in cords.split(',')][:3]
                        state[d][0] = x
                        state[d][1] = y

                    s = self.get_state_ind(state)
                    for a in range(self.n_actions):
                        s_a_map[s, a] = 0

        return s_a_map

    def parse_constraints_to_map(self, constraints):
        s_a_map = np.zeros((self.n_states, self.n_actions))

        for constraint in constraints:
            state = np.zeros((3, 2))
            for i in range(3):
                state[i][0] = constraint[2*i]
                state[i][1] = constraint[(2*i)+1]

            s = self.get_state_ind(state)
            a = constraint[-1]
            s_a_map[s, a] = 1

        return s_a_map

    # for model-free RL
    def step(self, state_ind, action_ind, check_if_valid=False):
        disk_to_move, peg_to = self.actions[action_ind]
        reward = 0
        done = False

        state = self.get_state_from_ind(state_ind)
        peg_from = state[disk_to_move][0]
        next_state = deepcopy(state)

        # if the disk is not clear nothing will happen
        if not self.is_clear(state, disk_to_move):
            return state_ind, reward, done

        disks_on_peg_to = self.get_disks_on_peg(state, peg_to)

        # check if no bigger disk is placed on a smaller disk
        if check_if_valid:
            for d in disks_on_peg_to:
                if d < disk_to_move:
                    return state_ind, reward, done

        if peg_from != peg_to:
            next_state[disk_to_move, 0] = peg_to
            next_state[disk_to_move, 1] = len(disks_on_peg_to)

        valid_move = True
        valid_move &= self._valid_state(state)
        valid_move &= self._valid_state(next_state)

        if (next_state == self.get_state_from_ind(self.objective)).all():
            reward = 100
            done = True

        return self.get_state_ind(next_state), reward, done

    def is_clear(self, state, disk):
        x, y = state[disk]

        for d in range(self.n_disks):
            if d != disk:
                d_x, d_y = state[d]
                if (x == d_x) and (y < d_y):
                    return False

        return True

    def get_disks_on_peg(self, state, peg):
        disks = []

        for d in range(self.n_disks):
            if state[d, 0] == peg:
                disks.append(d)

        return disks

    def _get_initial_state(self):
        state = np.zeros((self.n_disks, 2), dtype=int)

        for d in range(self.n_disks):
            state[d] = [0, d]

        return state

    def _get_goal_state(self):
        state = np.zeros((self.n_disks, 2), dtype=int)

        for d in range(self.n_disks):
            state[d] = [self.n_pegs-1, d]

        return state

    def _get_initial_state_probs(self):
        p = np.zeros((self.n_goals, self.n_states))

        for i in range(self.n_goals):
            p[i, self.initial_state_ind] = 1.0

        return p

    def _get_reward(self):
        r = np.full((self.n_goals, self.n_states, self.n_actions), 0)

        for i in range(self.n_goals):
            for a in range(self.n_actions):
                r[i, self.get_state_ind(self._get_goal_state())][a] = 1.0

        return r

    def _valid_state(self, state):
        # check if all disks have a different position
        for i in range(len(state)):
            for j in range(len(state)):
                if (i != j) and (state[i] == state[j]).all():
                    return False

        # check if no disk floats
        for disk in state:
            peg = disk[0]
            h = disk[1]

            if h > (len(self.get_disks_on_peg(state, peg)) - 1):
                return False

        return True

    def _transition_prob(self, s, s_, a):
        state = self.get_state_from_ind(s)
        state_ = self.get_state_from_ind(s_)
        next_state = deepcopy(state)

        disk_to_move, peg_to = self.actions[a]
        peg_from = state[disk_to_move][0]
        height_to = state_[disk_to_move][1]

        disks_on_peg_to = self.get_disks_on_peg(state, peg_to)
        disks_on_peg_from = self.get_disks_on_peg(
            state, peg_from)

        valid_move = True

        # check if current and next state are valid states
        valid_move &= self._valid_state(state)
        valid_move &= self._valid_state(state_)

        # check if there is a disk above the disk to move
        for d in disks_on_peg_from:
            if d != disk_to_move:
                if state[d][1] > state[disk_to_move][1]:
                    valid_move = False

        # check if there are no smaller disks on the goal peg
        if self.constrained:
            for d in disks_on_peg_to:
                # if there is a disk with a bigger index,
                # this means there is a smaller disk on the peg you want to move to, which means this is invalid
                if d > disk_to_move:
                    valid_move = False

        if peg_from == peg_to:
            valid_move = False

        if valid_move:
            next_state[disk_to_move, 0] = peg_to
            next_state[disk_to_move, 1] = len(disks_on_peg_to)

            if (state_ == next_state).all():
                return 1.0

        return 0.0
