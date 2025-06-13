from envs.sumo_unidir_3_way_junction import SumoUnidir3WayJunction
from envs.sumo_bidir_4_way_junction2 import SumoBidir4WayJunction2
from envs.sumo_unidir_4way_junction import SumoUnidir4WayJunction
from envs.sumo_combi import SumoCombi
from envs.sumo_combi2 import SumoCombi2
import trajectory as T
import argparse
import json
import numpy as np
import os
import pickle

from building_trajectories import build_trajectories
import random

def get_final_action(args, world, state):
    x, y, _ = world.decompose_state(state)
    gs = args.gridsize

    actions = np.array(world.actions_str)
    if (x == 1) and (y == 3):
        return np.where(actions == 'W')[0][0]
    if (x == 3) and (y == gs):
        return np.where(actions == 'N')[0][0]
    if (x == gs) and (y == 3):
        return np.where(actions == 'E')[0][0]
    if (x == 3) and (y == 1):
        return np.where(actions == 'S')[0][0]

    raise Exception('invalid final state')

def original_get_final_action(world, state):
    x, y, _ = world.decompose_state(state)

    actions = np.array(world.actions_str)
    if (x == 1) and (y == 3):
        return np.where(actions == 'W')[0][0]
    if (x == 3) and (y == 5):
        return np.where(actions == 'N')[0][0]
    if (x == 5) and (y == 3):
        return np.where(actions == 'E')[0][0]
    if (x == 3) and (y == 1):
        return np.where(actions == 'S')[0][0]

    raise Exception('invalid final state')


def parse_unidir_3way_junction(args, sim):
    world = SumoUnidir3WayJunction(sim, args)

    # create trajectories in the form of (s, a, s_)
    expert_trajectories = []
    for traj in sim['trajectories']:
        traj_ = []
        for i in range(len(traj) - 1):
            s = np.array(traj[i])
            s_ = np.array(traj[i + 1])
            a = np.where((world.actions == ((s_ - s)[:2])).all(axis=1))[0][0]

            x, y, l, f, r = s
            x_, y_, l_, f_, r_ = s_

            traj_.append([world.compose_state(x, y, r), a,
                          world.compose_state(x_, y_, r_)])

        expert_trajectories.append(T.Trajectory(traj_))

    return expert_trajectories


def original_parse_bidir_4way_junction2(args, sim):
    world = SumoBidir4WayJunction2(sim, args)

    # create trajectories in the form of (s, a, s_)
    expert_trajectories = []
    print(sim['trajectories'][-1])
    for traj in sim['trajectories']:
        traj_ = []
        for i in range(len(traj) - 1):
            s = np.array(traj[i])
            s_ = np.array(traj[i + 1])

            s_xy = np.array([int(cord) for cord in s[:2]])
            s_xy_ = np.array([int(cord) for cord in s_[:2]])

            a = np.where(
                (world.actions == (s_xy_[:2] - s_xy[:2])).all(axis=1))[0][0]

            x, y, tls = s
            x_, y_, tls_ = s_

            # check if no car violates the traffic lights
            if ((x == 2) and (y == 3) and (tls == 1) and (x_ == 3)) or \
                ((x == 4) and (y == 3) and (tls == 1) and (x_ == 3)) or \
                ((x == 3) and (y == 2) and (tls == 0) and (y_ == 3)) or \
                    ((x == 3) and (y == 4) and (tls == 0) and (y_ == 3)):
                print('traffic light violation in observation!')

            traj_.append([world.compose_state(int(x), int(y), int(tls)), a,
                          world.compose_state(int(x_), int(y_), int(tls))])

        traj_.append(
            [traj_[-1][-1], original_get_final_action(world, traj_[-1][-1]), traj_[-1][-1]])

        expert_trajectories.append(T.Trajectory(traj_))
    
    return expert_trajectories

def parse_bidir_4way_junction2(args, sim):
    """
    Rather than generating trajectories with the sim, we generate them using our build_Trajectories function.
    """
    print("Hello")
    world = SumoBidir4WayJunction2(sim, args)
    print(f"Redact North Value: {args.redact_north}")
    extended_expert_trajectories = build_trajectories(args, redact_north=args.redact_north)

    # create trajectories in the form of (s, a, s_)
    expert_trajectories = []
    v_count = 0 # Number of tls violation, for personal debugging

    # for traj in sim['trajectories']:
    for traj in extended_expert_trajectories:
        violation = False
        traj_ = []
        for i in range(len(traj) - 1):
            s = np.array(traj[i])
            s_ = np.array(traj[i + 1])

            s_xy = np.array([int(cord) for cord in s[:2]])
            s_xy_ = np.array([int(cord) for cord in s_[:2]])

            a = np.where(
                (world.actions == (s_xy_[:2] - s_xy[:2])).all(axis=1))[0][0]

            x, y, tls = s
            x_, y_, tls_ = s_

            # check if no car violates the traffic lights
            if ((x == 2) and (y == 3) and (tls == 1) and (x_ == 3)) or \
                ((x == 4) and (y == 3) and (tls == 1) and (x_ == 3)) or \
                ((x == 3) and (y == 2) and (tls == 0) and (y_ == 3)) or \
                    ((x == 3) and (y == 4) and (tls == 0) and (y_ == 3)):
                # print(f'traffic light violation! Initial: {traj[0]}, Goal: {traj[-1]}')
                violation = True
                continue

            traj_.append([world.compose_state(int(x), int(y), int(tls)), a,
                          world.compose_state(int(x_), int(y_), int(tls))])
        if violation:
            v_count += 1
            continue

        traj_.append(
            [traj_[-1][-1], get_final_action(args, world, traj_[-1][-1]), traj_[-1][-1]])

        expert_trajectories.append(T.Trajectory(traj_))
    
    print(f"{v_count} trajectories ommitted due to tls violations")
 
    return expert_trajectories

def parse_unidir_4way_junction(args, sim):
    world = SumoUnidir4WayJunction(sim, args)

    # create trajectories in the form of (s, a, s_)
    expert_trajectories = []
    for traj in sim['trajectories']:
        traj_ = []
        for i in range(len(traj) - 1):
            s = np.array(traj[i])
            s_ = np.array(traj[i + 1])
            a = np.where((world.actions == ((s_ - s)[:2])).all(axis=1))[0][0]

            x, y, l, f, r = s
            x_, y_, l_, f_, r_ = s_

            if ((x == 2) and (y == 3) and (r == 1) and (a == 1)) or \
                    ((x == 3) and (y == 4) and (r == 1) and (a == 3)):
                print(f'right priority violations: {s}')

            traj_.append([world.compose_state(x, y, r), a,
                          world.compose_state(x_, y_, r_)])

        traj_.append(
            [traj_[-1][-1], get_final_action(world, traj_[-1][-1]), traj_[-1][-1]])
        expert_trajectories.append(T.Trajectory(traj_))

    return expert_trajectories


def parse_combi(args, sim):
    world = SumoCombi(sim, args)

    expert_trajectories = []
    for traj in sim['trajectories']:
        traj_ = []
        for i in range(len(traj) - 1):
            s = np.array(traj[i])
            s_ = np.array(traj[i + 1])

            s_xy = np.array([int(cord) for cord in s[:2]])
            s_xy_ = np.array([int(cord) for cord in s_[:2]])

            a = np.where(
                (world.actions == (s_xy_[:2] - s_xy[:2])).all(axis=1))[0][0]

            x, y, tls, r = s
            x_, y_, tls_, r_ = s_

            # check if no car violates the traffic lights
            if ((x == 3) and (y == 4) and (tls == 1) and (x_ == 4)) or \
               ((x == 5) and (y == 4) and (tls == 1) and (x_ == 4)) or \
               ((x == 4) and (y == 3) and (tls == 0) and (y_ == 4)) or \
               ((x == 4) and (y == 5) and (tls == 0) and (y_ == 4)):
                print('traffic light violation in observation!')

            if ((x == 3) and (y == 4) and (r == 1) and (a == 4)) or \
                    ((x == 4) and (y == 5) and (r == 1) and (a == 3)) or \
                    ((x == 5) and (y == 4) and (r == 1) and (a == 2)) or \
                    ((x == 4) and (y == 3) and (r == 1) and (a == 1)):
                print(f'right priority violations: {s}')

            traj_.append([world.compose_state(x, y, tls, r), a,
                          world.compose_state(x_, y_, tls_, r_)])

        expert_trajectories.append(T.Trajectory(traj_))

    return expert_trajectories


def parse_combi2(args, sim):
    world = SumoCombi2(sim, args)

    expert_trajectories = []
    for traj in sim['trajectories']:
        traj_ = []
        for i in range(len(traj) - 1):
            s = np.array(traj[i])
            s_ = np.array(traj[i + 1])

            s_xy = np.array([int(cord) for cord in s[:2]])
            s_xy_ = np.array([int(cord) for cord in s_[:2]])

            a = np.where(
                (world.actions == (s_xy_[:2] - s_xy[:2])).all(axis=1))[0][0]

            x, y, tls, r = s
            x_, y_, tls_, r_ = s_

            if (x == 6) and (y == 5) and (tls == 0) and (a == 1):
                print('test')

            # check if no car violates the traffic lights
            if world.traffic_light_violation(world.compose_state(x, y, tls, r), a):
                print('traffic light violation in observation!')

            if world.right_priority_violation(world.compose_state(x, y, tls, r), a):
                print(f'right priority violations: {s}')

            traj_.append([world.compose_state(x, y, tls, r), a,
                          world.compose_state(x_, y_, tls_, r_)])

        expert_trajectories.append(T.Trajectory(traj_))

    return expert_trajectories


def main(args):

    # save expert trajectories
    result_path = f'results/{args.env}'
    file_path = f'{result_path}/run{args.run}_grid{args.gridsize}_o{args.num_observations}_rn{args.redact_north}_observations'

    if not os.path.exists(file_path):
        sim = json.load(open(f'sumo/{args.env}/{args.env}.json', 'r'))
        # If gridsize is 7, then run our alternative parse_observation code.

        if args.env == 'sumo_unidir_3way_junction':
            observations = parse_unidir_3way_junction(args, sim)
        elif args.env == 'sumo_bidir_4way_junction2' and args.gridsize == 5:
            observations = original_parse_bidir_4way_junction2(args, sim)
        elif args.env == 'sumo_bidir_4way_junction2' and args.gridsize != 5:
            observations = parse_bidir_4way_junction2(args, sim)
        elif args.env == 'sumo_unidir_4way_junction':
            observations = parse_unidir_4way_junction(args, sim)
        elif args.env == 'sumo_combi':
            observations = parse_combi(args, sim)
        elif args.env == 'sumo_combi2':
            observations = parse_combi2(args, sim)
        else:
            print(args.env, args.gridsize)
    

        if not os.path.exists(result_path):
            if not os.path.exists('results'):
                os.mkdir('results')
            os.mkdir(result_path)

        with open(f'{result_path}/run{args.run}_grid{args.gridsize}_o{args.num_observations}_rn{args.redact_north}_observations', 'wb') as file:
            pickle.dump(observations, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do stuff')

    parser.add_argument('--env', type=str, default='sumo_bidir_4way_junction2')
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--gridsize', type=int, default=5)
    parser.add_argument('--redact_north', type=int, default=0)
    parser.add_argument('--num_observations', type=int, default=30)
    parser.add_argument('--novel_candidate_elimination', type=bool, default=False)

    args = parser.parse_args()

    main(args)
