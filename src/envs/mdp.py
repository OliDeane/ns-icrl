import numpy as np
import os
from itertools import product
from tqdm import tqdm

class Mdp():
    def valid_state(self, s):
        return True

    def _transition_prob_table(self):
        print("Building Transition Prob Table")
        table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))

        # s1, s2, a = range(self.n_states), range(
        #     self.n_states), range(self.n_actions)
        # for s_from, s_to, a in product(s1, s2, a):
        for s_from in tqdm(range(self.n_states)):
            for s_to in range(self.n_states):
                for a in range(self.n_actions):
                    table[s_from, s_to, a] = self._transition_prob(s_from, s_to, a)

        return table

    def _get_p_transition_table_from_disk(self, env_name):
        if hasattr(self, 'constrained') and self.constrained:
            file_path = '{}_p_transition_c.npy'.format(env_name)
        else:
            file_path = '{}_p_transition.npy'.format(env_name)

        if not os.path.exists(file_path):
            p = self._transition_prob_table()
            np.save(file_path, p)
        else:
            p = np.load(file_path)

        return p
