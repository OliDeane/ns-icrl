from matplotlib import pyplot as plt
import numpy as np
import trajectory as T
import wandb
import time
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from tqdm import tqdm

def plot_heatmap(data):
    plt.figure(figsize=(24, 6))  # Increase the width from 8 to 12, height remains 6
    plt.imshow(data, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar()  # Show color scale
    plt.title('Heatmap of a n x m Array')
    plt.xlabel('Action')
    plt.ylabel('State (int)')
    # plt.show()

def visualize_array_as_2dgrid(args, world, input_arr, array_name="Visitation_Freq"):

    # Define the size of the grid
    grid_size_x, grid_size_y = (args.gridsize, args.gridsize)

    # Create a plot
    fig, ax = plt.subplots(figsize=(12, 10), nrows=1, ncols=2)  # Increased figure size
    
    for tls_val in [0,1]:
        ax[tls_val].set_xlim(0, grid_size_x)
        ax[tls_val].set_ylim(0, grid_size_y)

        # Draw gridlines
        ax[tls_val].set_xticks(np.arange(1, grid_size_x + 1))
        ax[tls_val].set_yticks(np.arange(1, grid_size_y + 1))
        ax[tls_val].grid(True)

        # Nomralize the array to be 0-1
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        # Apply min-max normalization
        if max_val != min_val and array_name=="Visitation_Freq":
            input_arr = (input_arr - min_val) / (max_val - min_val)



        # Define coordinates (you can change these to any values within the grid)
        # S is the current state and a is the action.
        for s in range(world.n_states):
            for a in range(world.n_actions):
                # print(s)
                value = input_arr[s,a]
                x,y,tls = world.decompose_state(s)
                # print(x,y,tls)
                if tls == tls_val:
                    actions_str = ['stay', 'S', 'E', 'N', 'W']
                    action = actions_str[a]

                    if action == "stay":
                        ax[tls_val].scatter(x - 0.15, y - 0.15, color="black", s=50*value)
                    else:
                        angle = action_to_angle(action)

                        dx = np.cos(angle) * value * 0.35  # Horizontal component of arrow, reduced size
                        dy = np.sin(angle) * value  * 0.35# Vertical component of arrow, reduced size

                        # ax[tls_val].arrow(x - 0.5, y - 0.5, dx, dy, head_width=0.05, head_length=0.05, fc="black", ec="black")  # Smaller arrow head
                        ax[tls_val].arrow(x - 0.5, y - 0.5, dx, dy, head_width=0.08*value, head_length=0.08*value, fc="black", ec="black", length_includes_head=True)  # Smaller arrow head


        filename=f"{args.run_name}_{array_name}_plot"
        # filename = "test_plot"

        # Label the axes
        ax[tls_val].set_xlabel('')
        ax[tls_val].set_ylabel('')
        ax[tls_val].set_title(args.run_name)
        ax[tls_val].set_title(f"{args.run_name} | {array_name} | TLS:{tls_val}")

        # Set the aspect of the plot to be equal
        ax[tls_val].set_aspect('equal', adjustable='box')
    plt.tight_layout()
    # Save and show the plot
    fig.savefig(f'pothole_output/{filename}.png')
    # plt.show()

def plot_heatmaps_depricated(array1, array2,iteration=0):
    """
    Display the difference between two arrays. 
    """

    # Calculate the difference array
    difference_array = np.equal(array1, array2).astype(int)  # Convert boolean results to integers (1 if equal, 0 otherwise)

    # Set up the matplotlib figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(36, 6))  # 1 row, 3 columns of plots

    # Plot the first array
    im0 = axs[0].imshow(array1, cmap='viridis', aspect='auto')
    fig.colorbar(im0, ax=axs[0])
    axs[0].set_title(f'Iteration {str(iteration)} | Policy T1')
    axs[0].set_xlabel('Action')
    axs[0].set_ylabel('State (int)')
    axs[0].set_xticks(np.arange(array1.shape[1]))
    axs[0].set_yticks(np.arange(array1.shape[0]))
    # axs[0].grid(which='both', color='white', linestyle='-', linewidth=0.5)  # Adjust grid color and line style here

    # Plot the second array
    im1 = axs[1].imshow(array2, cmap='viridis', aspect='auto')
    fig.colorbar(im1, ax=axs[1])
    axs[1].set_title(f'Iteration {str(iteration)} | Policy T2')
    axs[1].set_xlabel('Action')
    axs[1].set_ylabel('State')
    axs[1].set_xticks(np.arange(array1.shape[1]))
    axs[1].set_yticks(np.arange(array1.shape[0]))
    # axs[1].grid(which='both', color='white', linestyle='-', linewidth=0.5)

    # Plot the difference array
    im2 = axs[2].imshow(difference_array, cmap='viridis', aspect='auto')
    fig.colorbar(im2, ax=axs[2])
    axs[2].set_title('Difference Array (1=Equal, 0=Different)')
    axs[2].set_xlabel('Action')
    axs[2].set_ylabel('State (int)')
    axs[2].set_xticks(np.arange(array1.shape[1]))
    axs[2].set_yticks(np.arange(array1.shape[0]))
    # axs[2].grid(which='both', color='white', linestyle='-', linewidth=0.5)

    # Display the plot
    plt.tight_layout()  # Adjust layout to not overlap
    # plt.show()

def plot_heatmaps_interactive(world,array1, array2, iteration=0, array_name="Policy"):
    """
    Display the difference between two arrays using interactive plotly heatmaps.
    """
    # Calculate the difference array
    difference_array = np.equal(array1, array2).astype(int)  # Convert boolean results to integers (1 if equal, 0 otherwise)

    # Create subplots: 1 row, 3 columns
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f'Iteration {iteration} | {array_name} T1', 
                                                        f'Iteration {iteration} | {array_name} T2', 
                                                        'Difference Array (1=Equal, 0=Different)'])

    # Add heatmap for the first array
    fig.add_trace(go.Heatmap(z=array1, colorscale='Viridis'), row=1, col=1)

    # Add heatmap for the second array
    fig.add_trace(go.Heatmap(z=array2, colorscale='Viridis'), row=1, col=2)

    # Add heatmap for the difference array
    fig.add_trace(go.Heatmap(z=difference_array, colorscale='Viridis'), row=1, col=3)

    # Update layout
    fig.update_layout(height=1800, width=1800, title_text=f"Iteration {iteration} Heatmaps")

    # Update x-axes for all subplots
    fig.update_xaxes(title_text="Action", row=1, col=1)
    fig.update_xaxes(title_text="Action", row=1, col=2)
    fig.update_xaxes(title_text="Action", row=1, col=3)

    # Update y-axes for all subplots
    y_tick_labels = [world.decompose_state(s) for s in range(array1.shape[0])]
    x_tick_labels = [astr for astr in world.actions_str]
    # y_tick_labels = ["(x,y,tls)"] * array1.shape[0]
    fig.update_yaxes(title_text="State (int)", tickvals=list(range(len(y_tick_labels))), ticktext=y_tick_labels, row=1, col=1)
    fig.update_yaxes(title_text="State (int)", tickvals=list(range(len(y_tick_labels))), ticktext=y_tick_labels, row=1, col=2)
    fig.update_yaxes(title_text="State (int)", tickvals=list(range(len(y_tick_labels))), ticktext=y_tick_labels, row=1, col=3)
    fig.update_xaxes(title_text="Action", tickvals=list(range(len(x_tick_labels))), ticktext=x_tick_labels)

    # Display the plot
    fig.show()

def plot_heatmaps(array1, array2, iteration=0, plot_type="Policy"):
    """
    Display the difference between two arrays.
    """
    # Calculate the difference array
    difference_array = np.equal(array1, array2).astype(int)  # Convert boolean results to integers (1 if equal, 0 otherwise)

    # Set up the matplotlib figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(36, 6))  # 1 row, 3 columns of plots

    for i, (array, title) in enumerate(zip([array1, array2, difference_array], 
                                           [f'Iteration {str(iteration)} | {plot_type} T1', 
                                            f'Iteration {str(iteration)} | {plot_type} T2', 
                                            'Difference Array (1=Equal, 0=Different)'])):
        # Create a divider for the existing axes instance
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Plot the array
        im = axs[i].imshow(array, cmap='viridis', aspect='auto')
        fig.colorbar(im, cax=cax)
        axs[i].set_title(title)
        axs[i].set_xlabel('Action')
        axs[i].set_ylabel('State (int)')
        axs[i].set_xticks(np.arange(array.shape[1]))
        axs[i].set_yticks(np.arange(array.shape[0]))
        axs[i].tick_params(axis='y', labelsize=12)  # Adjust the fontsize for y-axis labels
        axs[i].invert_yaxis()  # Flip the y-axis

    # Display the plot
    plt.tight_layout()  # Adjust layout to not overlap
    # plt.show()

def plot_policy(data):
    # Create a figure with 2 subplots side by side

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 6))
    
    # Plot the first heatmap
    im1 = ax1.imshow(data[0,:,:], cmap='viridis', interpolation='nearest')
    fig.colorbar(im1, ax=ax1)  # Add a colorbar to the first subplot
    ax1.set_title("")
    ax1.set_xlabel('Action')
    ax1.set_ylabel('State (int)')
    
    # Plot the second heatmap
    im2 = ax2.imshow(data[1,:,:], cmap='viridis', interpolation='nearest')
    fig.colorbar(im2, ax=ax2)  # Add a colorbar to the second subplot
    ax2.set_title("")
    ax2.set_xlabel('Action')
    ax2.set_ylabel('State (int)')
    
    # Display the plot
    # plt.show()

def action_to_angle(action):
    if action == 'N':  # North - pointing up
        return np.pi / 2
    elif action == 'S':  # South - pointing down
        return -np.pi / 2
    elif action == 'E':  # East - pointing right
        return 0
    elif action == 'W':  # West - pointing left
        return np.pi
    return np.pi / 2  # Default to east if the action is unrecognized

def visualize_coordinates(args, world, trajectories, constraint_pairs=[], output_folder="pothole_output"):

    # Define coordinates (you can change these to any values within the grid)
    # S is the current state and a is the action.
    traj_coordinates = []
    for t in trajectories:
        for s, a, _ in t.transitions():
            sublst = [i for i in list(world.decompose_state(s))]
            sublst.append(a)

            if sublst not in traj_coordinates:
                traj_coordinates.append(sublst)
    
    # Define the size of the grid
    grid_size_x, grid_size_y = args.gridsize, args.gridsize

    # Create a plot
    fig, ax = plt.subplots(figsize=(args.gridsize, args.gridsize))  # Increased figure size
    ax.set_xlim(0, grid_size_x)
    ax.set_ylim(0, grid_size_y)

    # Draw gridlines
    ax.set_xticks(np.arange(1, grid_size_x + 1))
    ax.set_yticks(np.arange(1, grid_size_y + 1))
    ax.grid(True)

    # Add arrows at each coordinate in the trajectory list
    for x, y, tls, a in traj_coordinates:
        if tls == 0 and x == 3:
            color = 'red'
        elif tls == 0 and y == 3:
            color = 'green'
        elif tls == 1 and x == 3:
            color = 'green'
        elif tls == 1 and y == 3:
            color = 'red'
        else:
            color = 'blue'
        
        actions_str = ['stay', 'S', 'E', 'N', 'W']
        action = actions_str[a]

        if action == "stay" and color == 'red':
            ax.scatter(x - 0.25, y - 0.25, color=color, s=50)
        elif action == "stay" and color == 'green':
            ax.scatter(x - 0.75, y - 0.25, color=color, s=50)
        elif action == "stay" and color == 'blue':
            ax.scatter(x - 0.75, y - 0.25, color=color, s=50)
        else:
            angle = action_to_angle(action)
            dx = np.cos(angle) * 0.15  # Horizontal component of arrow, reduced size
            dy = np.sin(angle) * 0.15  # Vertical component of arrow, reduced size

            if color == "red":
                ax.arrow(x - 0.25, y - 0.25, dx, dy, head_width=0.05, head_length=0.05, fc=color, ec=color)  # Smaller arrow head
            elif color == "green":
                ax.arrow(x - 0.75, y - 0.25, dx, dy, head_width=0.05, head_length=0.05, fc=color, ec=color)  # Smaller arrow head
            else:
                ax.arrow(x - 0.75, y - 0.25, dx, dy, head_width=0.05, head_length=0.05, fc=color, ec=color)  # Smaller arrow head


    for c_sa in constraint_pairs:
        x, y, tls = c_sa[0]
        a = c_sa[1]

        if tls == 0 and x == 3:
            color = 'red'
        elif tls == 0 and y == 3:
            color = 'green'
        elif tls == 1 and x == 3:
            color = 'green'
        elif tls == 1 and y == 3:
            color = 'red'
        else:
            color = 'orange'

        actions_str = ['stay', 'S', 'E', 'N', 'W']
        action = actions_str[a]

        if action == "stay":
            if action == "stay" and color == 'red':
                ax.scatter(x - 0.25, y - 0.75, color=color, s=50)
            elif action == "stay" and color == 'green':
                ax.scatter(x - 0.75, y - 0.75, color=color, s=50)

        else:
            angle = action_to_angle(action)
            dx = np.cos(angle) * 0.15  # Horizontal component of arrow, reduced size
            dy = np.sin(angle) * 0.15  # Vertical component of arrow, reduced size
            
            if color == "red":
                ax.arrow(x - 0.25, y - 0.75, dx, dy, head_width=0.1, head_length=0.00, fc=color, ec=color)  # Smaller arrow head

            elif color == "green":
                ax.arrow(x - 0.75, y - 0.75, dx, dy, head_width=0.1, head_length=0.00, fc=color, ec=color)  # Smaller arrow head
            else:
                ax.arrow(x - 0.75, y - 0.75, dx, dy, head_width=0.1, head_length=0.00, fc=color, ec=color)  # Smaller arrow head

    # Set the title:
    # if not args.load_ilp_constraints and args.num_state_action_constraints == 9:
    #     title = "Initial Constraint Inference"
    #     filename = "initial_constraint_inference"
    # elif args.load_ilp_constraints and args.num_state_action_constraints == 9:
    #     title = "Generalised Constraints"
    #     filename = "generalised_constraints"
    # else:
    #     title = "Added Pothole Trajectories"
    #     filename = "pothole_trajectories"
    try:
        filename=f"{args.num_state_action_constraints}Constraints_{args.num_observations}Obs"
    except:
        filename=f"{args.run_name}_plot"

    # Label the axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    # ax.set_title(args.run_name)
    ax.set_title(filename)


    # Set the aspect of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    # Save and show the plot
    fig.savefig(f'{output_folder}/{filename}.png')
    plt.show()

def OLD_visualize_coordinates(world, trajectories, constraint_pairs=[]):

    # Define coordinates (you can change these to any values within the grid)
    traj_coordinates = []
    for t in trajectories:
        for s, a, _ in t.transitions():
            sublst = [i for i in list(world.decompose_state(s))]
            sublst.append(a)

            if sublst not in traj_coordinates:
                traj_coordinates.append(sublst)
    
    # coordinates = [(1, 1, 0, 1), (2, 2, 0, 2), (3, 3, 0, 3), (4, 4, 1, 2), (5, 5, 1, 1), (2, 3, 1,1), (4, 1, 0,0), (5, 2, 0,0)]

    # Define the size of the grid
    grid_size_x, grid_size_y = 7, 7

    # Create a plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, grid_size_x)
    ax.set_ylim(0, grid_size_y)

    # Draw gridlines
    ax.set_xticks(np.arange(1, grid_size_x + 1))
    ax.set_yticks(np.arange(1, grid_size_y + 1))
    ax.grid(True)

    # Add arrows at each coordinate in the trajectory list
    for x, y, tls, a in traj_coordinates:
        if tls == 0 and x==3:
            color = 'red'
            offset = 0
        elif tls == 0 and y==3:
            color='green'
            offset = 0
        elif tls == 1 and x==3:
            color = 'green'
            offset = 0
        elif tls == 1 and y == 3:
            color= 'red'
            offset = 0
        else:
            color='orange'
            offset = 0.0
        
        actions_str = ['stay', 'S', 'E', 'N', 'W']
        action = actions_str[a]

        if action == "stay":
            ax.scatter(x-0.5, y-0.5, color=color, s=100)
        else:
            angle = action_to_angle(action)
            dx = np.cos(angle) * 0.2  # Horizontal component of arrow
            dy = np.sin(angle) * 0.2  # Vertical component of arrow
            ax.arrow(x - 0.5 + offset, y - 0.5, dx, dy, head_width=0.1, head_length=0.1, fc=color, ec=color)


    for c_sa in constraint_pairs:
        x, y, tls = c_sa[0]
        a = c_sa[1]

        if tls == 0 and x==3:
            color = 'pink'
            offset = 0.1
        elif tls == 0 and y==3:
            color='blue'
            offset = 0.25
        elif tls == 1 and x==3:
            color = 'blue'
            offset = -0.1
        elif tls == 1 and y == 3:
            color= 'pink'
            offset = -0.25
        else:
            color='orange'
            offset = 0
        
        actions_str = ['stay', 'S', 'E', 'N', 'W']
        action = actions_str[a]

        if action == "stay":
            ax.scatter(x-0.5, y-0.5, color=color, s=100)
        else:
            angle = action_to_angle(action)
            dx = np.cos(angle) * 0.2  # Horizontal component of arrow
            dy = np.sin(angle) * 0.2  # Vertical component of arrow
            ax.arrow(x - 0.5, y - 0.5 + offset, dx, dy, head_width=0.1, head_length=0.1, fc=color, ec=color)



    # Label the axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    # Show the plot
    # plt.show()

def get_initial_probabilities_from_trajectories(world, trajectories):
    p = np.zeros(world.n_states * world.n_actions)

    for t in trajectories:
        t_0 = t.transitions()[0]
        p[world.compose_state_action_pair(t_0[0], t_0[1])] += 1.0

    return p / len(trajectories)

def get_initial_state_action_probabilities_from_trajectories(world, trajectories):
    p = np.zeros((world.n_states, world.n_actions))

    for t in trajectories:
        t_0 = t.transitions()[0]
        p[t_0[0], t_0[1]] += 1.0

    return p / len(trajectories)

def get_empirical_probability_distribution(world, trajectories):
    p = np.zeros((world.n_states, world.n_actions))
    n_transitions = 0

    for traj in trajectories:
        for tran in traj.transitions():
            p[tran[0], tran[1]] += 1
            n_transitions += 1

    return p / n_transitions

def get_goal_of_trajectory(world, trajectory):
    s, a, s_ = trajectory.transitions()[-1]
    s_d = world.decompose_state(s_)
    goal = np.where(
        np.array(list(map(lambda x: x == (s_d[0], s_d[1]), world.goals))))[0][0]
    return goal

def get_prob_of_demonstrations_under_policy(world, demonstrations, policy, p_initial):
    probs = []

    for trajectory in demonstrations:
        if world.n_goals > 1:
            goal = get_goal_of_trajectory(world, trajectory)
        else:
            goal = 0

        #p = p_initial[goal, trajectory.transitions()[0][0]]
        # probs.append(p)

        for s, a, s_ in trajectory.transitions():
            p = world.p_transition[int(s), int(s_), a] * policy[goal, int(s), a]
            probs.append(p)

    return np.mean(probs)

# def get_prob_of_demonstrations_under_policy(world, demonstrations, policy, p_initial):
#     probs = []
#
#     for trajectory in demonstrations:
#         probs.append(get_prob_of_trajectory_under_policy(
#             world, trajectory, policy, p_initial))
#
#     return np.mean(probs)
#
#
# def get_prob_of_trajectory_under_policy(world, trajectory, policy, p_initial):
#     p = p_initial[trajectory.transitions()[0][0]]
#
#     for s, a, s_ in trajectory.transitions():
#         p *= world.p_transition[s, s_, a] * policy[s, a]
#
#     return p


def softmax(x1, x2):
    x_max = np.maximum(x1, x2)
    x_min = np.minimum(x1, x2)
    return x_max + np.log(1.0 + np.exp(x_min - x_max))

def get_state_visit_freq(world, p_transition, p_action, p_initial, terminal, eps=1e-5):
    n_states, _, n_actions = p_transition.shape

    d = np.zeros(n_states)

    # set-up transition matrices for each action
    p_transition = np.copy(p_transition)
    p_transition = [np.array(p_transition[:, :, a])
                    for a in range(n_actions)]

    delta = np.inf
    while delta > eps:
        d_ = [[p_transition[a].T.dot(p_action[g, :, a] * d * world.valid_action[:, a])
               for g in range(world.n_goals)] for a in range(n_actions)]

        d_ = np.array(d_).sum(axis=0)
        d_ = (p_initial + d_).sum(axis=0)
        d_ = d_ / np.mean(d_)
        delta, d = np.max(np.abs(d_ - d)), d_

    d[np.array(terminal).flatten()] = 0

    return d

def get_state_action_visit_freq(args, world, p_transition, p_action, p_initial, terminal, eps=1e-5, iteration=0):
    
    # print(f"p_transition size: {p_transition.shape}")
    # print(f"p_transition sum: {sum(sum(p_transition))}")
    # np.set_printoptions(threshold=np.inf)
    # print(p_transition[(p_transition != 0) & (p_transition != 1)])
    # print(p_transition[(p_transition != 0) & (p_transition != 1)].shape)
    # print(p_transition)
    
    n_states, _, n_actions = p_transition.shape

    d_s = np.zeros(n_states)
    d_sa = np.zeros((n_actions, n_states))

    # set-up transition matrices for each action
    p_transition = np.copy(p_transition)
    p_transition = [np.array(p_transition[:, :, a])
                    for a in range(n_actions)]
    delta = np.inf

    # While not converged (ojd) - i.e., while the change in d_sa is < epsilon
    max_iterations=5000
    d_count = 0
    with tqdm(total=max_iterations) as pbar:
        while delta > eps and d_count < max_iterations:
            d_count+=1
            d_s = [[p_transition[a].T.dot(p_action[g, :, a] * d_s * world.valid_action[:, a])
                    for g in range(world.n_goals)] for a in range(n_actions)]
            d_s = np.array(d_s).sum(axis=0)
            d_s = p_initial + d_s

            d_sa_ = np.array([(d_s * p_action[:, :, a]).sum(axis=0) * world.valid_action[:, a]
                            for a in range(n_actions)])

            if d_count == 1:
                first_dsa = d_sa_

            # in the next iteration d_s should be of shape [n_states]
            d_s = d_s.sum(axis=0)
            d_s = d_s / np.mean(d_s)
            d_sa_ = d_sa_ / np.mean(d_sa_)

            delta, d_sa = np.max(np.abs(d_sa_ - d_sa)), d_sa_
    print(f"Visitation Frequency Iterations: {d_count}")
    # Plot the visitation frequency values
    # plot_heatmaps_interactive(world,first_dsa.T,d_sa.T, array_name="Visit Freqs", iteration=iteration)
    visualize_array_as_2dgrid(args,world,d_sa.T)

    if not args.add_terminal_states:
        # a terminal state cannot be selected as constraint
        d_s[np.array(terminal).flatten()] = 0
        for a in range(n_actions):
            d_sa[a, terminal] = 0.0
    
    # print(f"d_sa: {d_sa.shape}")
    # print(f"Flattened d_sa: {d_sa.flatten(order='F').shape}")
    return d_s, d_sa.flatten(order='F')

# Ask them this: when ou say M do you mean
def backward(world, objective, terminal):
    """
    Objective seems to be consistent throughout
    """
    n_states, _, n_actions = world.p_transition.shape

    p = [np.array(world.p_transition[:, :, a]) for a in range(n_actions)]
    er = np.exp(objective)
    zs = np.zeros(n_states)

    zs_ = []
    za_ = []
    # print(np.mean(objective))
    # plot_heatmap(er[3,:,:])
    for i in range(world.n_goals):
        # print(world.decompose_state_action_pair(terminal[i]))
        # print(f"Terminal: {terminal[i]}")
        # print(world.valid_action)

        # zs is a state partition function for a single goal.
        zs[terminal[i]] = 1.0

        for _ in tqdm(range(2 * n_states)):
            za = np.array([world.valid_action[:, a] * er[i, :, a] *
                           p[a].dot(zs) for a in range(n_actions)]).T

            # overflow protection
            za = za / np.mean(za)
            zs = za.sum(axis=1)
        zs_.append(zs)
        za_.append(za)

    zs_ = np.array(zs_)
    za_ = np.array(za_)
    backward_output = np.divide(za_, zs_[:, :, None], out=np.zeros_like(za_), where=zs_[:, :, None] != 0)
    # print(f"Backward Output Shape: {backward_output.shape}")
    return np.divide(za_, zs_[:, :, None], out=np.zeros_like(za_), where=zs_[:, :, None] != 0)

def is_state_action_constraint(world, candidate, trajectories, reject_unobserved_candidates=True):

    s_c, a_c = world.decompose_state_action_pair(candidate)

    # print(f"Trajectory: {[(world.decompose_state(s), world.actions_str[a]) for s,a,_ in trajectories[0].transitions()]}")
    # print(f"Canddidate: {world.decompose_state(s_c),world.actions_str[a_c]}")

    state_list = []
    for t in trajectories:
        for s,act,_ in t.transitions():
            state_list.append(s)
            # print((world.decompose_state(s), world.actions_str[act]))

    for t in trajectories:
        for s, a, _ in t.transitions():
            # print(world.decompose_state(s))
            if (s == s_c) and (a == a_c):
                # print(f"Candidate rejected with ORIGINAL condition: {world.decompose_state(s_c)},{world.actions_str[a]}")
                return False
    
    
    # Our new candidate elimination policy. If relevent flag is set to True,
    if s_c not in state_list and world.novel_candidate_elimination:
        # print([world.decompose_state(i) for i in state_list])
        # print(world.decompose_state(s_c))
        # print(f"Constraint found with NEW condition: {world.decompose_state(s_c)}")
        return False
            
    # print(f"Constraint found with ORIGINAL condition: {world.decompose_state(s_c)}, {world.actions_str[a_c]}")

    return True

def original_is_state_action_constraint(world, candidate, trajectories):
    s_c, a_c = world.decompose_state_action_pair(candidate)

    for t in trajectories:
        for s, a, _ in t.transitions():
            if (s == s_c) and (a == a_c):
                return False
    return True

def is_state_constraint(world, candidate, trajectories):
    if candidate in np.array(world.terminal).flatten():
        return False

    for t in trajectories:
        for s, a, _ in t.transitions():
            if (s == candidate):
                return False
    return True

def get_state_action_constraint(world, d, trajectories):
    """
    This goes through the state-action frequencies. Finds the maximum (i.e., most likely
    state-action pair according to the policy) and checks if it is the expert trajectopries (through 
    is_state_action_constraint). If it is, then the state-action frequency is set to 0, and the next most
    likely is checked. 
    is_state_action_constraint will return True if the candidate (s,a) constraint is not in the Trajectory set.
    """
    constraint_found = False

    while not constraint_found:
        c = np.argmax(d)

        if is_state_action_constraint(world, c, trajectories):
            constraint_found = True
        else:
            d[c] = 0.0

        if np.sum(d) == 0.0:
            return None, None

    return c, d[c]

def altered_get_state_action_constraint(world, d, trajectories):
    constraint_found = False

    while not constraint_found:
        c = np.argmax(d)

        if is_state_action_constraint(world, c, trajectories):
            constraint_found = True
        else:
            d[c] = 0.0

        if np.sum(d) == 0.0:
            return None, None

    return c, d[c]

def get_state_constraint(world, d, trajectories):
    constraint_found = False

    while not constraint_found:
        c = np.argmax(d)

        if is_state_constraint(world, c, trajectories):
            constraint_found = True
        else:
            d[c] = 0.0

        if np.sum(d) == 0.0:
            return None, None

    return c, d[c]

def kl_divergence(p, q):
    p = p.flatten()
    q = q.flatten()
    kl = 0

    for i in range(len(p)):
        kl += p[i] * np.log((p[i] + 0.00001)/(q[i] + 0.00001))

    return kl

def get_trajectories_from_policy(world, policy, initial, terminal):
    policy_exec = T.stochastic_policy_adapter(policy)
    trajectories = list(T.generate_trajectories(
        200, world, policy_exec, initial, terminal))

    return trajectories

def get_timing_table(vi, inference):
    columns = ['VI', 'Inference']
    table = [[vi, inference]]

    return wandb.Table(data=table, columns=columns)

def get_state_action_wandb_table(world, constraints):
    columns = world.feature_names
    table = []
    for c in constraints:
        table.append(world.convert_constraint_to_array(c))

    return wandb.Table(data=table, columns=columns)

def get_state_wandb_table(world, state_constraints):
    columns = world.feature_names[:-1]
    table = []
    for c in state_constraints:
        arr = world.convert_constraint_to_array((c, 0))[:-1]
        table.append(arr)

    return wandb.Table(data=table, columns=columns)

def add_empty_state_constraints(args, world, p_transition, trajectories):
    """ 
    This adds s,a pairs that lead to empty states as constraints. For each state s, for each action a
    and for each subsequent stats s';
    if the set of valid actions in s' are all 0 (no valid actions),
    and if it is possible to transfer from s to s' via a according to p_trans,
    and if (s,a) is not in T (i.,e., is_state_action_constraint returns True), then add (s,a) as a constraint.
    """

    all_empty_states_found = False

    while not all_empty_states_found:
        all_empty_states_found = True

        for s in range(world.n_states):
            for a in range(world.n_actions):
                for s_ in range(world.n_states):
                    # check if there are still valid actions in this state
                    if (p_transition[s, s_, a] > 0) and (np.sum(world.valid_action[s_]) == 0):
                        # check if constraint is not part of observations
                        c = world.compose_state_action_pair(s, a)
                        if is_state_action_constraint(world, c, trajectories):
                            if world.add_constraint(args, s, a):
                                all_empty_states_found = False

def add_connected_nodes(world, p_transition, reachable_states, s):
    for s_ in range(world.n_states):
        for a in world.get_valid_actions(s):
            if (p_transition[s, s_, a] > 0) and (s_ not in reachable_states):
                reachable_states.append(s_)

    return reachable_states

def add_unreachable_constraints(args, world, p_transition, p_initial):
    reachable_states = []

    for g in range(world.n_goals):
        for s in range(world.n_states):
            if np.sum(p_initial[g, s]) > 0:
                reachable_states.append(s)

    all_states_visited = False
    while not all_states_visited:
        reachable_states_ = reachable_states.copy()

        for s in reachable_states:
            reachable_states_ = add_connected_nodes(
                world, p_transition, reachable_states_, s)

        if len(reachable_states) == len(reachable_states_):
            all_states_visited = True
        else:
            reachable_states = reachable_states_.copy()

    # add all states which are not reachable as constraints
    for s in range(world.n_states):
        if s not in reachable_states:
            for a in world.get_valid_actions(s):
                world.add_constraint(args, s, a)

def moving_average(data, N=5):
    average = 0
    n = 0

    for i in range(len(data)-1, len(data)-1-N, -1):
        if i > -1:
            average += data[i]
            n += 1

    return average/n

def infer_state_constraints(args, world, trajectories, p_transition, eps=0.05, num_constraints=None):
    policies = []

    p_action = backward(world, world.objective, world.terminal)
    policies.append(p_action)

    delta_probs = []
    prob = get_prob_of_demonstrations_under_policy(
        world, trajectories, p_action, world.initial)

    if (num_constraints == None) and (eps == None):
        raise Exception(
            'infer_state_action_constraints: no stop condition specified')
    i = 0

    stop_condition = False

    while not stop_condition:
        d = get_state_visit_freq(
            world, p_transition, p_action, world.initial, world.terminal)

        # get constraint based on visitation frequencies and the expert trajectories
        constraint, _ = get_state_constraint(
            world, d, trajectories)

        # if 'get_state_action_constraint' returns None, this means
        # no more constraints can be found
        if constraint == None:
            return policies

        # add constraint to MDP
        world.add_state_constraint(args, constraint)

        # check for empty states and add to constraints
        #add_empty_state_constraints(args, world, p_transition, trajectories)

        if args.add_unreachable_states:
            add_unreachable_constraints(
                args, world, p_transition, world.initial)

        # update policy on MDP with the constraint added
        p_action = backward(world, world.objective, world.terminal)
        policies.append(p_action)

        prob_prev = prob
        prob = get_prob_of_demonstrations_under_policy(
            world, trajectories, p_action, world.initial)
        delta_prob = (prob - prob_prev) / \
            prob_prev if (prob_prev > 0) else 1
        delta_probs.append(delta_prob)

        if args.log:
            wandb.log({'n_state_constraints': i+1})
            wandb.log({'n_constraints': len(world.constraints)})
            wandb.log({'prob': prob})
            wandb.log({'delta_prob': delta_prob})
            wandb.log({'delta_prob_smooth': moving_average(delta_probs)})

        print('num state constraints: {} \r'.format(i+1), end='')
        i += 1

        if num_constraints == None:
            stop_condition = delta_prob <= eps
        else:
            stop_condition = i >= num_constraints

    print('')
    return policies

def infer_state_action_constraints(args, world, trajectories, p_transition, policies, eps=0.01, num_constraints=None):
    """
    This covers lines 2-17(?) in Algorithm 1 of the paper.
    """
    start = time.time()

    print(f"Objective:{[(world.decompose_state(tt[0]), world.decompose_state(tt[1])) for tt in world.terminal]}")
    p_action = backward(world, world.objective, world.terminal)
    vi_timing = time.time() - start

    # print(f"Backward Output Shape: {p_action.shape}")
    # print(f"Policies Len: {len(policies)}")
    # print("======================")
    policies.append(p_action)

    delta_probs = []
    prob = get_prob_of_demonstrations_under_policy(
        world, trajectories, p_action, world.initial)

    if (num_constraints == None) and (eps == None):
        raise Exception(
            'infer_state_action_constraints: no stop condition specified')
    i = 0

    stop_condition = False
    inference_timing = 0

    # The stop condition is considered true (i.e., stop the loop) if the number of constraints is inferred. In there setting, i is 9
    # so the loop repeats for 9 iterations. 

    while not stop_condition:
        # print("")
        # print("---------")
        # print("NEW CONSTRAINT INFERENCE ITERATION")
        # print("---------")
        p_action_temp = p_action

        # which state-action pairs occur in the nominal MDP when p_action is applied
        start = time.time()

        # This covers lines 10-12(?) in Algorithm 1
        d_s, d_sa = get_state_action_visit_freq(
            args, world, p_transition, p_action, world.initial, world.terminal, iteration=i)

        # print(f"VF Shape: {d_s.shape}")
        # This is line 14 in Algorithm 1
        constraint, _ = get_state_action_constraint(
            world, d_sa, trajectories)
        inference_timing += time.time() - start

        # if 'get_state_action_constraint' returns None, this means no more constraints can be found
        if constraint == None:
            print("NO MORE CONSTRAINTS FOUND")
            return policies, vi_timing, inference_timing

        # add constraint to MDP
        s_c, a_c = world.decompose_state_action_pair(constraint)
        world.add_constraint(args, s_c, a_c)

        # check for empty states and add to constraints (line 16 of Algorithm 1)
        add_empty_state_constraints(args, world, p_transition, trajectories)

        if args.add_unreachable_states:
            add_unreachable_constraints(
                args, world, p_transition, world.initial)

        # update policy on MDP with the constraint added (line 3-9(?) in Algorithm 1)
        p_action = backward(world, world.objective, world.terminal)


        # Plot changes to the policy
        # print("Changing states in policy:")
        # [print(f"{s}: {world.decompose_state(s)}") for s in range(128)]
        # plot_heatmaps_interactive(world, p_action_temp[1,:,:], p_action[1,:,:],iteration=i,array_name="Policy Goal 1")
        visualize_array_as_2dgrid(args,world,p_action[1,:,:], array_name="Policy")
        # visualize_coordinates(world, trajectories, world.constraints)
        
        # save constraint and policy
        policies.append(p_action)

        prob_prev = prob
        prob = get_prob_of_demonstrations_under_policy(
            world, trajectories, p_action, world.initial)
        delta_prob = (prob - prob_prev) / prob_prev
        delta_probs.append(delta_prob)

        if args.log:
            wandb.log({'n_state_action_constraints': i+1})
            wandb.log({'n_constraints': len(world.constraints)})
            wandb.log({'prob': prob})
            wandb.log({'delta_prob': delta_prob})
            wandb.log({'delta_prob_smooth': moving_average(delta_probs)})

        print('num state action constraints: {} \r'.format(i+1), end='')
        i += 1

        print("")
        if num_constraints == None:
            stop_condition = delta_prob <= eps
        else:
            stop_condition = i >= num_constraints

    return policies, vi_timing, inference_timing

def infer_dqn_state_action_constraints(args, world, trajectories, p_transition, policies, eps=0.01, num_constraints=None):
    """
    This covers lines 2-17(?) in Algorithm 1 of the paper.
    """
    print("Running the Infer State Action Constraints function")
    
    start = time.time()
    print("Running First backward Pass")
    p_action = backward(world, world.objective, world.terminal)
    vi_timing = time.time() - start

    # print(f"Policies Len: {len(policies)}")
    # print("======================")
    policies.append(p_action)

    delta_probs = []
    print("Runing the get_prob_of_demonstration function")
    prob = get_prob_of_demonstrations_under_policy(
        world, trajectories, p_action, world.initial)

    if (num_constraints == None) and (eps == None):
        raise Exception(
            'infer_state_action_constraints: no stop condition specified')
    i = 0

    stop_condition = False
    inference_timing = 0

    # The stop condition is considered true (i.e., stop the loop) if the number of constraints is inferred. In there setting, i is 9
    # so the loop repeats for 9 iterations. 
    print("Starting the constraint inference while loop...")
    print("----------------")
    while not stop_condition:
        # print("")
        # print("---------")
        # print("NEW CONSTRAINT INFERENCE ITERATION")
        # print("---------")
        p_action_temp = p_action

        # which state-action pairs occur in the nominal MDP when p_action is applied
        start = time.time()

        # This covers lines 10-12(?) in Algorithm 1
        print("Fetching state_action visit freq...")
        d_s, d_sa = get_state_action_visit_freq(
            args, world, p_transition, p_action, world.initial, world.terminal, iteration=i)

        # print(f"VF Shape: {d_s.shape}")
        # This is line 14 in Algorithm 1
        print("Fetching state_action constraint...")
        constraint, _ = get_state_action_constraint(
            world, d_sa, trajectories)
        inference_timing += time.time() - start

        # if 'get_state_action_constraint' returns None, this means no more constraints can be found
        if constraint == None:
            print("NO MORE CONSTRAINTS FOUND")
            return policies, vi_timing, inference_timing

        # add constraint to MDP
        print("Adding constraint to MDP...")
        s_c, a_c = world.decompose_state_action_pair(constraint)
        world.add_constraint(args, s_c, a_c)

        print("Checking for empty state constraints...")
        # check for empty states and add to constraints (line 16 of Algorithm 1)
        add_empty_state_constraints(args, world, p_transition, trajectories)

        print("Adding unreachable states...")
        if args.add_unreachable_states:
            add_unreachable_constraints(
                args, world, p_transition, world.initial)

        # update policy on MDP with the constraint added (line 3-9(?) in Algorithm 1)
        print("Updating policy with new constraint (Backward Pass)...")
        backtime1 = time.time()
        p_action = backward(world, world.objective, world.terminal)
        backtime2 = time.time()
        print(f"Backward Pass Time: {backtime2-backtime1}")

        # Plot changes to the policy
        # print("Changing states in policy:")
        # [print(f"{s}: {world.decompose_state(s)}") for s in range(128)]
        # plot_heatmaps_interactive(world, p_action_temp[1,:,:], p_action[1,:,:],iteration=i,array_name="Policy Goal 1")
        print("Visualizing Array as 2d grid...")
        visualize_array_as_2dgrid(args,world,p_action[1,:,:], array_name="Policy")
        # visualize_coordinates(world, trajectories, world.constraints)
        
        # save constraint and policy
        policies.append(p_action)

        print("Fetching prob of demos under policy...")
        prob_prev = prob
        prob = get_prob_of_demonstrations_under_policy(
            world, trajectories, p_action, world.initial)
        delta_prob = (prob - prob_prev) / prob_prev
        delta_probs.append(delta_prob)

        if args.log:
            wandb.log({'n_state_action_constraints': i+1})
            wandb.log({'n_constraints': len(world.constraints)})
            wandb.log({'prob': prob})
            wandb.log({'delta_prob': delta_prob})
            wandb.log({'delta_prob_smooth': moving_average(delta_probs)})

        print('num state action constraints: {} \r'.format(i+1), end='')
        i += 1

        print("")
        if num_constraints == None:
            stop_condition = delta_prob <= eps
        else:
            stop_condition = i >= num_constraints

    return policies, vi_timing, inference_timing

def save_learned_constraints_to_json(args,world):
    
    # save the constraints to a json file
    lc_filename = 'pothole_output/learned_constraints.json'
    with open(lc_filename, 'r') as json_file:
        l_c = json.load(json_file)

    print(world.constraints)
    l_c[args.eta] = []

    with open(lc_filename, 'w') as json_file:
        json.dump(l_c, json_file)

def infer_constraints(args, world, trajectories, dqn_pass=False):
    if args.num_observations > 0:
        trajectories = trajectories[:args.num_observations]
    
    policies = []
    vi_timing = 0
    inference_timing = 0
    total_t1 = time.time()

    if dqn_pass:
        policies, vi_timing, inference_timing = infer_dqn_state_action_constraints(
                args, world, trajectories, world.p_transition, policies, args.delta_p_sa, args.num_state_action_constraints)
        
    else:
        # delta_p_s always seems to be None. It is used as an epsilon value in the infer_state_action_constraints function.
        if (args.delta_p_s != None) or ((args.num_state_constraints != None) and (args.num_state_constraints > 0)):
            print('infer state constraints')
            policies = infer_state_constraints(
                args, world, trajectories, world.p_transition, args.delta_p_s, args.num_state_constraints)

        if (args.delta_p_sa != None) or ((args.num_state_action_constraints != None) and (args.num_state_action_constraints > 0)):
            print('infer state-action constraints')
            policies, vi_timing, inference_timing = infer_state_action_constraints(
                args, world, trajectories, world.p_transition, policies, args.delta_p_sa, args.num_state_action_constraints)
    total_t2 = time.time()
    print(f"State-action Infereence Time: {total_t2-total_t1}")
    # print(f"DELTA_P_SA: {args.delta_p_sa}")

    # This is taking all state, action pairs from the trajectories and filtering out duplicates. These are all
    # considered valid state-actions.
    # It is logging the valid_state_actions in the log file.

    valid_states = []
    valid_state_actions = []
    for trajectory in trajectories:
        for s, a, s_ in trajectory.transitions():
            state = world.decompose_state(s)

            if state not in valid_states:
                valid_states.append(state)
            if (state, a) not in valid_state_actions:
                valid_state_actions.append((state, a))
    
    # world.constraints contains all the constraints inferred so far. 
    state_action_constraints = world.constraints
    state_constraints = world.get_state_constraints()
    
    # print("state-action constraints:")
    # print(len(world.constraints))
    # print(world.constraints)
    # visualize_coordinates(world, trajectories, world.constraints)

    # Policies are updated each time a new constraint is found (through the backward pass function). 
    # It is added to the policies list. We then seem to take the final policy (with all constraints considered)
    # as the output policy. 
    policies = policies[-1]
    print(f"Backward Pass Time: {vi_timing}")
    print(f"Inference Timing: {inference_timing}")
    
    if args.log:
        wandb.log({'constraints': get_state_action_wandb_table(
            world, state_action_constraints)})
        wandb.log({'state_constraints': get_state_wandb_table(
            world, state_constraints)})
        wandb.log({'valid_states': get_state_wandb_table(
            world, valid_states)})
        wandb.log({'valid_state_actions': get_state_action_wandb_table(
            world, valid_state_actions)})
        wandb.log({'timings': get_timing_table(vi_timing, inference_timing)})

        for g in range(world.n_goals):
            wandb.log({'policy_{}'.format(g): wandb.Table(
                data=policies[g], columns=world.actions_str)})
        
        # save_learned_constraints_to_json(args, world)


    return state_action_constraints, state_constraints, policies

def infer_constraints_for_divergent_comparison(args, world, trajectories, dqn_pass=False):

    """
    This is their original method for constraint inference. I just remove logging
    """
    if args.num_observations > 0:
        trajectories = trajectories[:args.num_observations]
    
    policies = []
    vi_timing = 0
    inference_timing = 0
    total_t1 = time.time()

    if dqn_pass:
        policies, vi_timing, inference_timing = infer_dqn_state_action_constraints(
                args, world, trajectories, world.p_transition, policies, args.delta_p_sa, args.num_state_action_constraints)
        
    else:
        # delta_p_s always seems to be None. It is used as an epsilon value in the infer_state_action_constraints function.
        if (args.delta_p_s != None) or ((args.num_state_constraints != None) and (args.num_state_constraints > 0)):
            print('infer state constraints')
            policies = infer_state_constraints(
                args, world, trajectories, world.p_transition, args.delta_p_s, args.num_state_constraints)

        if (args.delta_p_sa != None) or ((args.num_state_action_constraints != None) and (args.num_state_action_constraints > 0)):
            print('infer state-action constraints')
            policies, vi_timing, inference_timing = infer_state_action_constraints(
                args, world, trajectories, world.p_transition, policies, args.delta_p_sa, args.num_state_action_constraints)
    total_t2 = time.time()
    print(f"State-action Infereence Time: {total_t2-total_t1}")
    # print(f"DELTA_P_SA: {args.delta_p_sa}")

    # This is taking all state, action pairs from the trajectories and filtering out duplicates. These are all
    # considered valid state-actions.
    # It is logging the valid_state_actions in the log file.

    valid_states = []
    valid_state_actions = []
    for trajectory in trajectories:
        for s, a, s_ in trajectory.transitions():
            state = world.decompose_state(s)

            if state not in valid_states:
                valid_states.append(state)
            if (state, a) not in valid_state_actions:
                valid_state_actions.append((state, a))
    
    # world.constraints contains all the constraints inferred so far. 
    state_action_constraints = world.constraints
    state_constraints = world.get_state_constraints()

    # Policies are updated each time a new constraint is found (through the backward pass function). 
    # It is added to the policies list. We then seem to take the final policy (with all constraints considered)
    # as the output policy. 
    policies = policies[-1]
    print(f"Backward Pass Time: {vi_timing}")
    print(f"Inference Timing: {inference_timing}")
    
    # if args.log:
    #     wandb.log({'constraints': get_state_action_wandb_table(
    #         world, state_action_constraints)})
    #     wandb.log({'state_constraints': get_state_wandb_table(
    #         world, state_constraints)})
    #     wandb.log({'valid_states': get_state_wandb_table(
    #         world, valid_states)})
    #     wandb.log({'valid_state_actions': get_state_action_wandb_table(
    #         world, valid_state_actions)})
    #     wandb.log({'timings': get_timing_table(vi_timing, inference_timing)})

    #     for g in range(world.n_goals):
    #         wandb.log({'policy_{}'.format(g): wandb.Table(
    #             data=policies[g], columns=world.actions_str)})
        
        # save_learned_constraints_to_json(args, world)


    return state_action_constraints, state_constraints, policies
