import argparse
import json
import numpy as np
import xml.etree.ElementTree as ET
import os
import pathlib
import math


def get_scale_coef(x_min, x_max, y_min, y_max):
    a = (y_max - y_min)/(x_max - x_min)
    b = (a * x_min) + y_min

    return a, b


def get_scaling(args, nodes):
    min_ = np.min(nodes)
    max_ = np.max(nodes)
    min_quant = 0
    max_quant = (max_ - min_) / args.quant_step
    a, b = get_scale_coef(min_, max_, min_quant, max_quant)

    return min_quant, max_quant, a, b


def parse_trajectories_unidir_3way_junction(vehicles_per_time_step):
    trajectories = {}

    for t in range(len(vehicles_per_time_step)):
        vehicles = vehicles_per_time_step[t]

        for vehicle_i in vehicles:
            i_id, i_x, i_y = vehicle_i
            if i_id not in trajectories.keys():
                trajectories[i_id] = []

            i_x = min(i_x, 6)
            i_y = min(i_y, 4)

            # check if there is a vehicle j in proximity of i
            l, f, r = 0, 0, 0
            for vehicle_j in vehicles:
                j_id, j_x, j_y = vehicle_j

                if i_id != j_id:
                    if j_x == (i_x + 1):
                        if j_y == (i_y + 1):
                            l = 1
                        elif j_y == i_y:
                            f = 1
                        elif j_y == (i_y - 1):
                            r = 1

            trajectories[i_id].append([i_x, i_y, l, f, r])

    return trajectories


def parse_trajectories_unidir_3way_junction_fol(vehicles_per_time_step):
    # parse to first-order logic (FOL) representation
    trajectories = {}

    for t in range(len(vehicles_per_time_step)):
        vehicles = vehicles_per_time_step[t]

        for vehicle_i in vehicles:
            i_id, i_x, i_y = vehicle_i
            if i_id not in trajectories.keys():
                trajectories[i_id] = []

            before_junction = 0

            if (i_x == 5) and (i_y == 4):
                road = 0
            elif (i_y == 4) and (i_x < 5):
                road = 1
                if i_x == 4:
                    before_junction = 1
            elif (i_x == 5) and (i_y < 4):
                road = 2
                if i_y == 3:
                    before_junction = 1
            elif (i_y == 4) and (i_x > 5):
                road = 3

            # check if there is a vehicle j in proximity of i
            l, f, r = 0, 0, 0

            for vehicle_j in vehicles:
                j_id, j_x, j_y = vehicle_j

                if i_id != j_id:
                    if j_x == (i_x + 1):
                        if j_y == (i_y + 1):
                            l = 1
                        elif j_y == i_y:
                            f = 1
                        elif j_y == (i_y - 1):
                            r = 1

            trajectories[i_id].append([road, before_junction, l, f, r])

    return trajectories


def parse_trajectories_bidir_3way_junction(vehicles_per_time_step):
    trajectories = {}

    for t in range(len(vehicles_per_time_step)):
        vehicles_at_t = vehicles_per_time_step[t]

        for vehicle_i in vehicles_at_t:
            i_id, i_x, i_y = vehicle_i
            if i_id not in trajectories.keys():
                trajectories[i_id] = []

            # check if there is a vehicle j in proximity of i
            r, r_j_id, goal, goal_r = 0, 0, 0, 0
            for vehicle_j in vehicles_at_t:
                j_id, j_x, j_y = vehicle_j

                if i_id != j_id:
                    # i comes from down and j comes from the right
                    if (i_x == 3) and (i_y == 1):
                        if (j_x == 4) and (j_y == 3):
                            r = 1
                            r_j_id = j_id

                    # i comes from the left and j comes from down
                    elif (i_x == 1) and (i_y == 2):
                        if (j_x == 3) and (j_y == 1):
                            r = 1
                            r_j_id = j_id

            trajectories[i_id].append([i_x, i_y, r, r_j_id, goal, goal_r])

    goals = []

    for vehicle in trajectories.keys():
        trajectory = trajectories[vehicle]
        goal = trajectory[-1][:2]

        if not (goal in goals):
            goals.append(goal)

    def get_goal_index(x, y):
        # left
        if (x == 0) and (y == 3):
            return 0
        # right
        elif ((x == 5) or (x == 6)) and (y == 2):
            return 1
        # down
        elif (x == 2) and (y == 0):
            return 2

    # set goal for each vehicle
    for vehicle in trajectories.keys():
        goal = get_goal_index(
            trajectories[vehicle][-1][0], trajectories[vehicle][-1][1])

        if goal is None:
            raise Exception("goal not specified for x={}, y={}".format(
                trajectories[vehicle][-1][0], trajectories[vehicle][-1][1]))

        for i in range(len(trajectories[vehicle])):
            # set the goal of the current vehicle
            trajectories[vehicle][i][4] = goal

            # set the goal of the vehicle to the right
            if trajectories[vehicle][i][2] != 0:
                vehicle_r = trajectories[vehicle][i][3]
                trajectories[vehicle][i][5] = get_goal_index(
                    trajectories[vehicle_r][-1][0], trajectories[vehicle_r][-1][1])

    return trajectories


def parse_trajectories_bidir_4way_junction(vehicles_per_time_step, tls_per_time_step):
    trajectories = {}

    for t in range(len(vehicles_per_time_step)):
        vehicles = vehicles_per_time_step[t]
        tls = tls_per_time_step[t]

        for vehicle_i in vehicles:
            id, x, y = vehicle_i

            # clip x and y coordinates
            x = min(x, 5)
            y = min(y, 5)

            if id not in trajectories.keys():
                trajectories[id] = []

            trajectories[id].append([x, y, tls, 0])

    def get_goal_index(x, y):
        # left
        if (x == 0) and (y == 3):
            return 0
        # right
        elif (x == 5) and (y == 2):
            return 1
        # down
        elif (x == 2) and (y == 0):
            return 2
        # up
        elif (x == 3) and (y == 5):
            return 3

        return None

    for traj in trajectories:
        for i in range(len(trajectories[traj])):
            goal = get_goal_index(
                trajectories[traj][-1][0], trajectories[traj][-1][1])

            if goal is None:
                raise Exception("goal not specified for x={}, y={}".format(
                    trajectories[traj][-1][0], trajectories[traj][-1][1]))

            trajectories[traj][i][3] = goal
    return trajectories


def parse_trajectories_bidir_4way_junction2(vehicles_per_time_step, tls_per_time_step):
    trajectories = {}

    for t in range(len(vehicles_per_time_step)):
        vehicles = vehicles_per_time_step[t]
        tls = tls_per_time_step[t]

        for vehicle_i in vehicles:
            id, x, y = vehicle_i

            # clip x and y coordinates (1-5)
            x = max(min(x, 5), 1)
            y = max(min(y, 5), 1)

            if id not in trajectories.keys():
                trajectories[id] = []

            trajectories[id].append([x, y, tls])

    return trajectories


def parse_trajectories_combi(vehicles_per_time_step, tls_per_time_step):
    trajectories = {}

    for t in range(len(vehicles_per_time_step)):
        vehicles = vehicles_per_time_step[t]
        tls = tls_per_time_step[t]

        for vehicle_i in vehicles:
            i_id, i_x, i_y = vehicle_i

            # clip x and y coordinates (1-5)
            i_x = max(min(i_x, 7), 1)
            i_y = max(min(i_y, 7), 1)

            if i_id not in trajectories.keys():
                trajectories[i_id] = []

            # check if there is a vehicle j in proximity of i
            r = 0
            for vehicle_j in vehicles:
                j_id, j_x, j_y = vehicle_j

                if i_id != j_id:
                    if (i_x == 4) and (i_y == 5) and (j_x == 5) and (j_y == 6):
                        r = 1
                    if (i_x == 3) and (i_y == 4) and (j_x == 2) and (j_y == 5):
                        r = 1
                    elif (i_x == 5) and (i_y == 4) and (j_x == 6) and (j_y == 3):
                        r = 1
                    elif (i_x == 4) and (i_y == 3) and (j_x == 3) and (j_y == 2):
                        r = 1

            trajectories[i_id].append([i_x, i_y, tls, r])

    return trajectories


def parse_trajectories_unidir_4way_junction(vehicles_per_time_step):
    trajectories = {}

    for t in range(len(vehicles_per_time_step)):
        vehicles = vehicles_per_time_step[t]

        for vehicle_i in vehicles:
            i_id, i_x, i_y = vehicle_i
            if i_id not in trajectories.keys():
                trajectories[i_id] = []

            i_x = max(min(i_x, 5), 1)
            i_y = max(min(i_y, 5), 1)

            # check if there is a vehicle j in proximity of i
            l, f, r = 0, 0, 0
            for vehicle_j in vehicles:
                j_id, j_x, j_y = vehicle_j

                if i_id != j_id:
                    if (i_x == 2) and (i_y == 3) and (j_x == 3) and (j_y == 2):
                        r = 1
                    elif (i_x == 3) and (i_y == 4) and (j_x == 2) and (j_y == 3):
                        r = 1

            trajectories[i_id].append([i_x, i_y, l, f, r])

    return trajectories


def parse_trajectories_combi2(vehicles_per_time_step, tls_per_time_step):
    trajectories = {}

    for t in range(len(vehicles_per_time_step)):
        vehicles = vehicles_per_time_step[t]
        tls = tls_per_time_step[t]

        for vehicle_i in vehicles:
            i_id, i_x, i_y = vehicle_i
            if i_id not in trajectories.keys():
                trajectories[i_id] = []

            i_x = max(min(i_x, 7), 1)
            i_y = max(min(i_y, 7), 1)

            # check if there is a vehicle j in proximity of i
            r = 0
            for vehicle_j in vehicles:
                j_id, j_x, j_y = vehicle_j

                if i_id != j_id:
                    if (i_x == 2) and (i_y == 4) and (j_x == 3) and (j_y == 3):
                        r = 1

            trajectories[i_id].append([i_x, i_y, tls, r])

    return trajectories


def main(args):
    # path to /sumo/env
    path = '{}/{}'.format(pathlib.Path(__file__).parent.resolve(), args.env)

    # parse node information
    nodes = ET.parse(path + '/{}.nod.xml'.format(args.env)).getroot()

    x_min, x_max, x_a, x_b = get_scaling(
        args, [float(node.attrib['x']) for node in nodes])
    y_min, y_max, y_a, y_b = get_scaling(
        args, [float(node.attrib['y']) for node in nodes])

    # parse trajectories
    sim = ET.parse(path + '/sim.xml').getroot()

    vehicles_per_time_step = []

    for time_step in sim:
        vehicles = []
        for vehicle in time_step:
            if (args.env == 'sumo_unidir_4way_junction') or \
                    (args.env == 'sumo_combi2'):
                x = round((x_a * float(vehicle.attrib['x'])) + x_b)
                y = round((y_a * float(vehicle.attrib['y'])) + y_b)
            else:
                x = math.floor((x_a * float(vehicle.attrib['x'])) + x_b)
                y = math.floor((y_a * float(vehicle.attrib['y'])) + y_b)

            if (args.env == 'sumo_bidir_4way_junction2') or \
                    (args.env == 'sumo_combi'):
                def map_coordinate(x):
                    return (x+1)//2

                x = map_coordinate(x)
                y = map_coordinate(y)

            vehicles.append([vehicle.attrib['id'], x, y])
        vehicles_per_time_step.append(vehicles)

    if args.env == 'sumo_unidir_3way_junction':
        if args.fol:
            trajectories = parse_trajectories_unidir_3way_junction_fol(
                vehicles_per_time_step)
        else:
            trajectories = parse_trajectories_unidir_3way_junction(
                vehicles_per_time_step)
    elif args.env == 'sumo_bidir_3way_junction':
        trajectories = parse_trajectories_bidir_3way_junction(
            vehicles_per_time_step)
    elif (args.env == 'sumo_bidir_4way_junction') or \
            (args.env == 'sumo_bidir_4way_junction2') or \
            (args.env == 'sumo_combi') or \
            (args.env == 'sumo_combi2'):
        sim_a = ET.parse(path + '/sim_add.xml').getroot()

        tls_per_time_step = []
        for time_step in sim_a:
            if time_step.attrib['state'][0] == 'r':
                tls_per_time_step.append(0)
            else:
                tls_per_time_step.append(1)

        if args.env == 'sumo_bidir_4way_junction':
            trajectories = parse_trajectories_bidir_4way_junction(
                vehicles_per_time_step, tls_per_time_step)
        elif args.env == 'sumo_bidir_4way_junction2':
            trajectories = parse_trajectories_bidir_4way_junction2(
                vehicles_per_time_step, tls_per_time_step)
        elif args.env == 'sumo_combi':
            trajectories = parse_trajectories_combi(
                vehicles_per_time_step, tls_per_time_step)
        elif args.env == 'sumo_combi2':
            trajectories = parse_trajectories_combi2(
                vehicles_per_time_step, tls_per_time_step)
    elif args.env == 'sumo_unidir_4way_junction':
        trajectories = parse_trajectories_unidir_4way_junction(
            vehicles_per_time_step)
        # get possible actions from trajectories
    actions = []

    if args.fol:
        actions.append('stay')
        actions.append('straight')
        actions.append('right')
    else:
        for trajectory in list(trajectories.values()):
            for i in range(len(trajectory) - 1):
                a = (np.array(trajectory[i+1][:2]) -
                     np.array(trajectory[i][:2])).tolist()
                if a not in actions:
                    actions.append(a)

    # get initial and goal states
    initial = []
    goal = []

    for trajectory in list(trajectories.values()):
        if trajectory[0] not in initial:
            initial.append(trajectory[0])
        if trajectory[-1] not in goal:
            goal.append(trajectory[-1])

    out = {}
    out['x_min'] = x_min
    out['x_max'] = x_max
    out['y_min'] = y_min
    out['y_max'] = y_max
    out['actions'] = actions
    out['initial'] = initial
    out['goal'] = goal
    out['trajectories'] = list(trajectories.values())

    with open('{}/{}{}.json'.format(path, args.env, '_fol' if args.fol else ''), 'w') as file:
        json.dump(out, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        default="sumo_combi")
    parser.add_argument('--quant_step', type=int, default=10)
    parser.add_argument('--fol', type=int, default=0)

    args = parser.parse_args()

    main(args)
