import argparse
import json
import os

def get_constraint_files_inds(file_names):
    s_c_i = [i for i, s in enumerate(
        file_names) if 'state_constraints_' in s]
    if len(s_c_i) != 0:
        s_c_i = s_c_i[0]
    else:
        s_c_i = None

    c_i = [i for i, s in enumerate(
        file_names) if ('constraints_' in s) and ('state constraints_' not in s)]
    if len(c_i) != 0:
        c_i = c_i[0]
    else:
        c_i = None

    v_s_a_i = [i for i, s in enumerate(
        file_names) if 'valid_state_actions_' in s]
    if len(v_s_a_i) != 0:
        v_s_a_i = v_s_a_i[0]
    else:
        v_s_a_i = None

    v_s_i = [i for i, s in enumerate(
        file_names) if 'valid_states_' in s]
    if len(v_s_i) != 0:
        v_s_i = v_s_i[0]
    else:
        v_s_i = None

    policy_inds = [i for i, s in enumerate(
        file_names) if 'policy' in s]

    return s_c_i, c_i, v_s_a_i, v_s_i, policy_inds

def read_constraint_files(env=None, experiment=None, from_results=False):
    dir = {}

    if from_results:
        if (env == None) or (experiment == None):
            raise Exception(
                'if "from_results" is true, "env" and "experiment" should be specified')
        dir_path = f'results/{env}/{experiment}'

        constraint_file = json.load(
            open(f'{dir_path}/constraints.json', 'r'))
        state_constraint_file = json.load(
            open(f'{dir_path}/state_constraints.json', 'r'))
        valid_state_actions_file = json.load(
            open(f'{dir_path}/valid_state_actions.json', 'r'))
        valid_state_file = json.load(
            open(f'{dir_path}/valid_states.json', 'r'))

        dir.update({'constraints': constraint_file})
        dir.update({'state_constraints': state_constraint_file})
        dir.update({'valid_state_actions': valid_state_actions_file})
        dir.update({'valid_states': valid_state_file})
        dir.update({'policies': None})  # TODO

    else:
        dir_path = 'wandb/latest-run/files/media/table/'

        file_names = sorted(os.listdir(dir_path))

        s_c_i, c_i, v_s_a_i, v_s_i, policy_inds \
            = get_constraint_files_inds(file_names)

        if c_i != None:
            constraint_file = json.load(open(dir_path + file_names[c_i], 'r'))
            dir.update({'constraints': constraint_file})

        if s_c_i != None:
            state_constraint_file = json.load(
                open(dir_path + file_names[s_c_i], 'r'))
            dir.update({'state_constraints': state_constraint_file})

        if v_s_a_i != None:
            valid_state_actions_file = json.load(
                open(dir_path + file_names[v_s_a_i], 'r'))
            dir.update({'valid_state_actions': valid_state_actions_file})

        if v_s_i != None:
            valid_state_file = json.load(
                open(dir_path + file_names[v_s_i], 'r'))
            dir.update({'valid_states': valid_state_file})

        policies = []
        for i in range(len(policy_inds)):
            policy_file = json.load(
                open(dir_path + file_names[policy_inds[i]], 'r'))
            policies.append(policy_file)
        dir.update({'policies': policies})

    return dir

def sumo_generate_acuity_files(args, valid_state_actions, constraint_state_actions):

    """
    Example Input:
    Valid State Actions:[([1, 3, 0], 0), ([1, 3, 0], 2), ([2, 3, 0], 0), ([2, 3, 0], 2), ([3, 3, 0], 0), ([3, 3, 0], 2), ([4, 3, 0], 2), ([5, 3, 0], 0), ([5, 3, 0], 2)]
    Constraint State Actions: [((3, 2, 0), 3)]
    """

    pos_example_str = 'sample(n{}). at(n{},{},{}). tls(n{},{}). go(n{}'
    neg_example_str = 'sample(p{}). at(p{},{},{}). tls(p{},{}). go(p{}'
    with open(f'prolog_files/{args.env}/{args.env}.bk', 'w') as output_file:
        with open(f'prolog_files/{args.env}/{args.env}.f', 'w') as f_file:
            with open(f'prolog_files/{args.env}/{args.env}.n', 'w') as n_file:
                # valid state action pairs
                i = 0

                acuity_neg_examples = []
                for state_action in valid_state_actions:
                    state,action = state_action[:-1], state_action[-1]
                    example_str = parse_bidir_4way_junction2_example_for_acuity(
                        pos_example_str, i, state, action)
                    output_file.write(example_str)
                    n_file.write(f'invalid(n{i}).\n')
                    i += 1
                    acuity_neg_examples.append(example_str)
                output_file.write('\n')

                # state action constraints
                i = 0
                acuity_pos_examples = []
                for state_action in constraint_state_actions:
                    state,action = state_action[:-1], state_action[-1]
                    example_str = parse_bidir_4way_junction2_example_for_acuity(
                        neg_example_str, i, state, action)
                    output_file.write(example_str)
                    f_file.write(f'invalid(p{i}).\n')
                    acuity_pos_examples.append(example_str)

                    # Add again to allow for induction without single rule problem.
                    example_str = parse_bidir_4way_junction2_example_for_acuity(
                        neg_example_str, "0"+str(i), state, action)
                    output_file.write(example_str)
                    f_file.write(f'invalid(p{i}).\n')
                    acuity_pos_examples.append(example_str)

                    i += 1
                output_file.write('\n')

                
    
    return acuity_neg_examples, acuity_pos_examples
    
def parse_bidir_4way_junction2_example_for_acuity(sub_str, i, state, action):
    # sub_str = sub_str.format(i, i, state[0], state[1], i, '1', i)


    if state[2]:
        # sub_str += ', tls(,1)'
        sub_str = sub_str.format(i, i, state[0], state[1], i, '1', i)
    else:
        sub_str = sub_str.format(i, i, state[0], state[1], i, '0', i)
        # sub_str += ', tls0'

    if action == 0:
        sub_str += ', zero)'
    elif action == 1:
        sub_str += ', south)'
    elif action == 2:
        sub_str += ', east)'
    elif action == 3:
        sub_str += ', north)'
    elif action == 4:
        sub_str += ', west)'

    sub_str += '. \n'

    return sub_str

def bidir_4way_junction2(args, output_file, state_constraint_file, constraint_file, valid_state_action_file):
    pos_example_str = '#pos(p{}, {{at({},{})'
    neg_example_str = '#neg(n{}, {{at({},{})'

    # valid state action pairs
    i = 0
    for state_action in valid_state_action_file['data']:
        state = state_action[:-1]
        action = state_action[-1]



def main(args):
    # files = read_constraint_files(
    #     env=args.env, experiment=f'run{args.run}_c{args.num_constraints}_o{args.num_observations}', from_results=True)
    
    # Read in the divergent trajectory
    folder = 'divergence_output/results'
    filename = f'grid{args.gridsize}_o{args.num_observations}_c{args.num_state_action_constraints}'

    with open(f'{folder}/{filename}.json', 'r') as json_file:
        divergent_constraint_data = json.load(json_file)

    if args.env == 'sumo_bidir_4way_junction2':
        # bidir_4way_junction2(args, "output_file", files['state_constraints'], files['constraints'], files['valid_state_actions'])
        valid_state_action_data = divergent_constraint_data['valid_state_actions']
        constraint_data = divergent_constraint_data['state_action_constraints']
        sumo_generate_acuity_files(args, valid_state_action_data, constraint_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do stuff')

    parser.add_argument('--env', type=str,
                        default='sumo_bidir_4way_junction2')
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--num_constraints', type=int, default=5)
    parser.add_argument('--num_state_action_constraints', type=int, default=1)
    parser.add_argument('--num_observations', type=int, default=0)
    parser.add_argument('--gridsize', type=int, default=5)
    parser.add_argument('--latest', type=bool, default=True)

    args = parser.parse_args()
    main(args)