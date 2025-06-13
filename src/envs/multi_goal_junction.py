from envs.simple_junction import SimpleJunction
import numpy as np


class MultiGoalJunction(SimpleJunction):
    def __init__(self, size, constrained=False):
        SimpleJunction.__init__(self, size, constrained)

        self.goals = {'bottom': [self.compose_state(x=self.n_x // 2, y=0, tl=i) for i in range(2**self.n_tl)],
                      'top': [self.compose_state(x=self.n_x // 2, y=self.n_y - 1, tl=i) for i in range(2**self.n_tl)],
                      'left': [self.compose_state(x=0, y=self.n_y // 2, tl=i) for i in range(2**self.n_tl)],
                      'right': [self.compose_state(x=self.n_x - 1, y=self.n_y // 2, tl=i) for i in range(2**self.n_tl)]}

        # we overwrite the methods which set initial state-action probs, rewards and terminal states
        # we also added a check that these initializations are only executed for the base class
        # that is why we still have to do them here
        self.terminal = self.goals
        self.reward = self._get_reward()
        self.initial = self._get_initial_state_probs()
        self.objective = self.reward

    def augment_with_constraints(self):
        # add constraint on all state not on the road and the bottom traffic light
        SimpleJunction.augment_with_constraints(self)

        # stop before traffic light comming from the
        # top
        self.add_constraint(self.compose_state(
            self.n_x//2, (self.n_y//2) + 1, 1), 4)
        # left
        self.add_constraint(self.compose_state(
            (self.n_x//2) - 1, self.n_y//2, 0), 1)
        # right
        self.add_constraint(self.compose_state(
            (self.n_x//2) + 1, self.n_y//2, 0), 3)

    def _get_initial_state_probs(self):
        goals_k = self.goals.keys()
        initial = {'bottom': np.zeros(self.n_states),
                   'top': np.zeros(self.n_states),
                   'left': np.zeros(self.n_states),
                   'right': np.zeros(self.n_states)}

        # the initial states corresponding to some goal state are all goal states except itself
        for key_i in goals_k:
            for key_j in goals_k:
                if key_j != key_i:
                    for goal in self.goals[key_j]:
                        initial[key_i][goal] = 1.0 / 6

        return initial

    def _get_reward(self):
        self.num_rewards = len(self.terminal)
        rewards = {'bottom': np.zeros((self.n_states, self.n_actions)),
                   'top': np.zeros((self.n_states, self.n_actions)),
                   'left': np.zeros((self.n_states, self.n_actions)),
                   'right': np.zeros((self.n_states, self.n_actions))}

        for goal_k in self.goals:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    if s in self.terminal[goal_k]:
                        rewards[goal_k][s][a] = 1.0

        return rewards

    def get_initial_state_action_prob(self):
        initial_sa_prob = {'bottom': np.zeros((self.n_states, self.n_actions)),
                           'top': np.zeros((self.n_states, self.n_actions)),
                           'left': np.zeros((self.n_states, self.n_actions)),
                           'right': np.zeros((self.n_states, self.n_actions))}

        for goal_k in self.goals:
            for s in range(self.n_states):
                if self.initial[goal_k][s] > 0:
                    # the validity of an action does not depend on the goal
                    # constraints apply for all goals
                    valid_actions = self.get_valid_actions(s)
                    prob = 1.0 / len(valid_actions)

                    for a in valid_actions:
                        initial_sa_prob[goal_k][int(s), a] = prob

        return initial_sa_prob

    def get_empty_goal_dict(self):
        return {'bottom': None,
                'top': None,
                'left': None,
                'right': None}
