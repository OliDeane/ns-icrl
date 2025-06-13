from envs.mdp import Mdp
import numpy as np


class ConstrainedMdp(Mdp):
    def get_valid_actions(self, state):
        """
        Get the possible actions given a state
        """

        valid_actions = []
        for i in range(self.n_actions):
            if self.valid_action[int(state), i] == 1:
                valid_actions.append(i)

        return np.array(valid_actions)

    def get_initial_state_action_prob(self):
        initial_sa_prob = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            if self.initial[s] > 0:
                valid_actions = self.get_valid_actions(s)
                prob = 1.0 / len(valid_actions)

                for a in valid_actions:
                    initial_sa_prob[int(s), a] = prob

        return initial_sa_prob

    def add_constraint(self, args, s, a):
        
        if not self.valid_state(s):
            return False

        if not args.add_terminal_states:
            if s in np.array(self.terminal).flatten():
                return False

        self.valid_action[s, a] = 0

        c = (self.decompose_state(s), a)
        if c not in self.constraints:
            self.constraints.append(c)
            return True

        return False

    def add_state_constraint(self, args, s_constraint):
        result = True

        # add all state-action pairs where the state equals s_constraint
        for a in range(self.n_actions):
            result &= self.add_constraint(args, s_constraint, a)

        # add all state-action pairs which produce s_constraint as constraints
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if self.p_transition[s, s_constraint, a] > 0:
                    result &= self.add_constraint(args, s, a)

        return result

    def get_state_constraints(self):
        state_constraints = list(
            np.where(np.sum(self.valid_action, axis=1) == 0)[0])

        for i in range(len(state_constraints)):
            state_constraints[i] = tuple(
                self.decompose_state(state_constraints[i]))

        return state_constraints
