import numpy as np
import random
import argparse

def get_compass_position(coords):
    if coords[1] > 3:
        return "N"
    elif coords[1] < 3:
        return "S"
    elif coords[0] > 3:
        return "E"
    elif coords[0] < 3:
        return "W"
    else:
        return "center"

def at_junction(coords, initial_compass_pos):
    if initial_compass_pos == "N" and coords[1] == 4:
        return True
    elif initial_compass_pos == "S" and coords[1] == 2:
        return True
    elif initial_compass_pos == "W" and coords[0] == 4:
        return True
    elif initial_compass_pos == "E" and coords[0] == 2:
        return True
    else:
        return False

def select_regular_move_nopothole(s, initial_compass_pos, goal_compass_pos, stay_prob=2/3):
        '''
        This can be merged with the original function. I add it here to avoid insertig buigs unecessarily.
        '''
        sx,sy,st = s
    
        car_compass_position = get_compass_position([sx,sy])
        car_at_junction = at_junction([sx,sy], initial_compass_pos)
        
        if initial_compass_pos == "N":

            if car_compass_position == "N":
                if car_at_junction:
                    if st == 0:
                        action_as_str = 'stay'
                    else:
                        action_as_str = random.choices(['stay', 'S'], [stay_prob,1-stay_prob])[0]
                else:
                    # We are in the north and not at a junction. We are travelling away from the north.
                    action_as_str = random.choices(['stay', 'S'], [stay_prob,1-stay_prob])[0]
            
            else:
                # move towards the goal
                action_as_str = random.choices(['stay', goal_compass_pos], [stay_prob,1-stay_prob])[0]      

        elif initial_compass_pos == "S":

            # Otheriwse, continue towards the goal accounting for the tls if at teh junction
            if car_compass_position == initial_compass_pos:
                if car_at_junction:
                    if st == 0:
                        action_as_str = 'stay'
                    else:
                        action_as_str = random.choices(['stay', 'N'], [stay_prob,1-stay_prob])[0]
                else:
                    action_as_str = random.choices(['stay', 'N'], [stay_prob,1-stay_prob])[0]
            
            else:
                # move towards the goal
                action_as_str = random.choices(['stay', goal_compass_pos], [stay_prob,1-stay_prob])[0]
    
        elif initial_compass_pos == "W":


            # Otheriwse, continue towards the goal accounting for the tls if at teh junction
            if car_compass_position == initial_compass_pos:
                if car_at_junction:
                    if st == 1:
                        action_as_str = 'stay'
                    else:
                        action_as_str = random.choices(['stay', 'E'], [stay_prob,1-stay_prob])[0]
                else:
                    action_as_str = random.choices(['stay', 'E'], [stay_prob,1-stay_prob])[0]
            
            else:
                # move towards the goal
                action_as_str = random.choices(['stay', goal_compass_pos], [stay_prob,1-stay_prob])[0]
        
        
        elif initial_compass_pos == "E":

            # Otheriwse, continue towards the goal accounting for the tls if at teh junction
            if car_compass_position == initial_compass_pos:
                if car_at_junction:
                    if st == 1:
                        action_as_str = 'stay'
                    else:
                        action_as_str = random.choices(['stay', 'W'], [stay_prob,1-stay_prob])[0]
                else:
                    action_as_str = random.choices(['stay', 'W'], [stay_prob,1-stay_prob])[0]
            
            else:
                # move towards the goal
                action_as_str = random.choices(['stay', goal_compass_pos], [stay_prob,1-stay_prob])[0]
        
        return action_as_str

def select_pothole_move(s, initial_compass_pos, goal_compass_pos, stay_prob=2/3):
        '''
        This can be merged with the original function. I add it here to avoid insertig buigs unecessarily.
        '''
        sx,sy,st = s
    
        car_compass_position = get_compass_position([sx,sy])
        car_at_junction = at_junction([sx,sy], initial_compass_pos)
        
        if initial_compass_pos == "N":

            if car_compass_position == "N":
                if car_at_junction:
                    if st == 0:
                        action_as_str = 'stay'
                    else:
                        action_as_str = random.choices(['stay', 'S'], [stay_prob,1-stay_prob])[0]
                else:
                    # We are in the north and not at a junction. We are travelling away from the north. And we want to avoid the pothole.
                    if sx==3 and sy==7: 
                        # If before pothole, travel to the side to avoid it
                        action_as_str = random.choices(['stay', 'E'], [stay_prob,1-stay_prob])[0]
                    elif sx==4 and sy==5: 
                        # If after the pothole, travel back on to the road
                        action_as_str = random.choices(['stay', 'W'], [stay_prob,1-stay_prob])[0]
                    else:
                        # Otherwise continue going south
                        action_as_str = random.choices(['stay', 'S'], [stay_prob,1-stay_prob])[0]
            
            else:
                # move towards the goal
                action_as_str = random.choices(['stay', goal_compass_pos], [stay_prob,1-stay_prob])[0]      

        elif initial_compass_pos == "S":
            
            # Check if we are around the pothole
            if sx==3 and sy==5: 
                # If before pothole, travel to the side to avoid it
                action_as_str = random.choices(['stay', 'W'], [stay_prob,1-stay_prob])[0]
            elif sx==2 and sy==7: 
                # If after the pothole, travel back on to the road
                action_as_str = random.choices(['stay', 'E'], [stay_prob,1-stay_prob])[0]

            # Otheriwse, continue towards the goal accounting for the tls if at teh junction
            elif car_compass_position == initial_compass_pos:
                if car_at_junction:
                    if st == 0:
                        action_as_str = 'stay'
                    else:
                        action_as_str = random.choices(['stay', 'N'], [stay_prob,1-stay_prob])[0]
                else:
                    action_as_str = random.choices(['stay', 'N'], [stay_prob,1-stay_prob])[0]
            
            else:
                # move towards the goal
                action_as_str = random.choices(['stay', goal_compass_pos], [stay_prob,1-stay_prob])[0]
    
        elif initial_compass_pos == "W":

            # Check if we are around the pothole
            if sx==3 and sy==5: 
                # If before pothole, travel to the side to avoid it
                action_as_str = random.choices(['stay', 'W'], [stay_prob,1-stay_prob])[0]
            elif sx==2 and sy==7: 
                # If after the pothole, travel back on to the road
                action_as_str = random.choices(['stay', 'E'], [stay_prob,1-stay_prob])[0]

            # Otheriwse, continue towards the goal accounting for the tls if at the junction
            elif car_compass_position == initial_compass_pos:
                if car_at_junction:
                    if st == 1:
                        action_as_str = 'stay'
                    else:
                        action_as_str = random.choices(['stay', 'E'], [stay_prob,1-stay_prob])[0]
                else:
                    action_as_str = random.choices(['stay', 'E'], [stay_prob,1-stay_prob])[0]
            
            else:
                # move towards the goal
                action_as_str = random.choices(['stay', goal_compass_pos], [stay_prob,1-stay_prob])[0]
        
        
        elif initial_compass_pos == "E":

            # Check if we are around the pothole
            if sx==3 and sy==5: 
                # If before pothole, travel to the side to avoid it
                action_as_str = random.choices(['stay', 'W'], [stay_prob,1-stay_prob])[0]
            elif sx==2 and sy==7: 
                # If after the pothole, travel back on to the road
                action_as_str = random.choices(['stay', 'E'], [stay_prob,1-stay_prob])[0]

            # Otheriwse, continue towards the goal accounting for the tls if at teh junction
            elif car_compass_position == initial_compass_pos:
                if car_at_junction:
                    if st == 1:
                        action_as_str = 'stay'
                    else:
                        action_as_str = random.choices(['stay', 'W'], [stay_prob,1-stay_prob])[0]
                else:
                    action_as_str = random.choices(['stay', 'W'], [stay_prob,1-stay_prob])[0]
            
            else:
                # move towards the goal
                action_as_str = random.choices(['stay', goal_compass_pos], [stay_prob,1-stay_prob])[0]
        
        return action_as_str



def select_move(s, initial_compass_pos, goal_compass_pos, stay_prob=2/3):
        
        sx,sy,st = s
    
        car_compass_position = get_compass_position([sx,sy])
        car_at_junction = at_junction([sx,sy], initial_compass_pos)
        
        if initial_compass_pos == "N":

            if car_compass_position == "N":
                if car_at_junction:
                    if st == 0:
                        action_as_str = 'stay'
                    else:
                        action_as_str = random.choices(['stay', 'S'], [stay_prob,1-stay_prob])[0]
                else:
                    action_as_str = random.choices(['stay', 'S'], [stay_prob,1-stay_prob])[0]
            
            else:
                # move towards the goal
                action_as_str = random.choices(['stay', goal_compass_pos], [stay_prob,1-stay_prob])[0]      

        elif initial_compass_pos == "S":

            if car_compass_position == initial_compass_pos:
                if car_at_junction:
                    if st == 0:
                        action_as_str = 'stay'
                    else:
                        action_as_str = random.choices(['stay', 'N'], [stay_prob,1-stay_prob])[0]
                else:
                    action_as_str = random.choices(['stay', 'N'], [stay_prob,1-stay_prob])[0]
            
            else:
                # move towards the goal
                action_as_str = random.choices(['stay', goal_compass_pos], [stay_prob,1-stay_prob])[0]
    
        elif initial_compass_pos == "W":

            if car_compass_position == initial_compass_pos:
                if car_at_junction:
                    if st == 1:
                        action_as_str = 'stay'
                    else:
                        action_as_str = random.choices(['stay', 'E'], [stay_prob,1-stay_prob])[0]
                else:
                    action_as_str = random.choices(['stay', 'E'], [stay_prob,1-stay_prob])[0]
            
            else:
                # move towards the goal
                action_as_str = random.choices(['stay', goal_compass_pos], [stay_prob,1-stay_prob])[0]
        
        
        elif initial_compass_pos == "E":

            if car_compass_position == initial_compass_pos:
                if car_at_junction:
                    if st == 1:
                        action_as_str = 'stay'
                    else:
                        action_as_str = random.choices(['stay', 'W'], [stay_prob,1-stay_prob])[0]
                else:
                    action_as_str = random.choices(['stay', 'W'], [stay_prob,1-stay_prob])[0]
            
            else:
                # move towards the goal
                action_as_str = random.choices(['stay', goal_compass_pos], [stay_prob,1-stay_prob])[0]
        
        return action_as_str

def build_trajectory(initial_state, goal):
    
    # actions = [[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]]
    # actions_str = ['stay', 'S', 'E', 'N', 'W']
    
    action_dict = {'stay':[0, 0], 'S':[0, -1], 'E':[1, 0], 'N':[0, 1], 'W':[-1, 0]}
    s = initial_state

    initial_compass_pos = get_compass_position(initial_state[0:2])
    goal_compass_pos = get_compass_position(goal[0][0:2])

    stay_prob = 2/3
    traj = [s]

    while s not in goal:
        
        # Select the move based on the current state s
        action_as_str = select_move(s, initial_compass_pos, goal_compass_pos, stay_prob)

        # Make the move
        a = action_dict[action_as_str]
        
        s[-1] = random.randint(0,1) # change the tls value
        s = [x + y for x, y in zip(s, a + [0] * (len(s) - len(a)))]

        # s_ = s.copy()
        traj.append(s.copy())

    return traj

def build_pothole_trajectory(initial_state, goal, stay_prob=0.66666):
    
    # actions = [[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]]
    # actions_str = ['stay', 'S', 'E', 'N', 'W']
    action_dict = {'stay':[0, 0], 'S':[0, -1], 'E':[1, 0], 'N':[0, 1], 'W':[-1, 0]}
    s = initial_state

    initial_compass_pos = get_compass_position(initial_state[0:2])
    goal_compass_pos = get_compass_position(goal[0][0:2])

    traj = [s]

    while s not in goal:
        
        # Select the move based on the current state s
        action_as_str = select_pothole_move(s, initial_compass_pos, goal_compass_pos, stay_prob)
        # action_as_str = select_regular_move_nopothole(s, initial_compass_pos, goal_compass_pos, stay_prob)


        # Make the move
        a = action_dict[action_as_str]
        
        s[-1] = random.randint(0,1) # change the tls value
        s = [x + y for x, y in zip(s, a + [0] * (len(s) - len(a)))]

        traj.append(s.copy())

    return traj

def build_trajectories(args, num_observation_iterations = 37, redact_north=False):

    gs = args.gridsize
    if redact_north:
        # Leave out top portion of the grid
        initial_states = [[1,3,0],[3,1,0],[gs,3,0]]
        goals = [[[1,3,0],[1,3,1]], [[3,1,0],[3,1,1]], [[gs,3,0],[gs,3,1]]]

    else:
        # Include north portion of grid. Implement wirh pothole trajectories
        initial_states = [[1,3,0],[3,1,0],[3,gs,0],[gs,3,0]]
        goals = [[[1,3,0],[1,3,1]], [[3,1,0],[3,1,1]], [[3,gs,0],[3,gs,1]], [[gs,3,0],[gs,3,1]]]

    
    observations = []

    # _ = build_trajectory(initial_states[1],goals[0])
    for _ in range(num_observation_iterations):
        for goal in goals:
            for i_s in initial_states:
                if i_s not in goal:
                    observations.append(build_pothole_trajectory(i_s,goal))

    return observations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='do stuff')
    parser.add_argument('--gridsize', type=int, default=5)
    args = parser.parse_args()
    print(build_trajectories())
        
