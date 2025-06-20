import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from PIL import Image
import os
import json

def read_local_invalid_state_actions():
    
    """
    Objective: Read in the invalid constraint-action pairs generated by ACUITY. Combine with those
    generated from the previous constraint loop.
    This function is just for testing the DQL and is build for reading from the local directory.
    """

    result_path = "divergence_output/results/"
    # acuity_output_filename = f'{result_path}grid{args.gridsize}_o{args.num_observations}_c{args.num_state_action_constraints}_invalidSAs.txt'
    constraint_filename = f'{result_path}grid10_o30_c1.json'
    acuity_output_filename = f'onRoad.txt'


    invalid_state_action_lst = []
    actions_str_prolog = ['zero', 'south', 'east', 'north', 'west']
    if os.path.exists(acuity_output_filename):
        with open(acuity_output_filename, 'r') as file:
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace
                if line:  # Ensure line is not empty
                    ll = line.split(',')
                    invalid_state_action_lst.append([int(ll[0][1:]),
                                                    int(ll[1]),
                                                    int(ll[2]),
                                                    actions_str_prolog.index(ll[3][:-1])])
            
    if os.path.exists(constraint_filename):
        with open(constraint_filename, 'r') as file:
            divergent_constraint_data = json.load(file)
        constraint_data = divergent_constraint_data['state_action_constraints']
        invalid_state_action_lst = invalid_state_action_lst + constraint_data
    
    return invalid_state_action_lst

def original_paper_plot_state(player_x, player_y, tls, rwd, target_x=None, target_y=None, size=5, runtype_label="Constraint Learning"):
    
    # if target_x is None or target_y is None:
    #     target_x, target_y = (size-1, int(size/2))

    # middle_value_list = [int(size/2)] * size
    middle_value_list = [3] * size
    road_x_list = [i for i in middle_value_list] + list(range(0, size))
    road_y_list = list(range(0, size)) + [i for i in middle_value_list]
    road2_x_list = middle_value_list + list(range(0, size))
    road2_y_list = list(range(0, size)) + middle_value_list

    env = np.zeros((size, size, 3), dtype=np.uint8)  # starts an rbg of our size

    for rx1, ry1, rx2, ry2 in list(zip(road_x_list, road_y_list, road2_x_list, road2_y_list)):  # Generate the road
        env[size-1-ry1][rx1] = (100, 100, 100)
        env[size-1-ry2][rx2] = (100, 100, 100)
    
    env[size-1][:] = 255
    env[:, 0] = 255

    env[size-1-target_y][target_x] = (255, 175, 0)
    env[size-1-player_y][player_x] = (0, 255, 0)

    # Display TLS:
    env[size-1][0] = (0, 0, 255) if tls == 0 else (0, 255, 0)
    
    # Display pothole
    if size >= 10:
        env[4][3] = (100, 0, 100)

    # Display with cv2
    img = Image.fromarray(env, 'RGB').resize((300, 300), Image.NEAREST)
    img_array = np.array(img)

    # Define the text properties
    text = str(rwd)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)
    thickness = 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate text position for top right corner
    text_x = img_array.shape[1] - text_size[0] - 10  # 10 pixels from the right side
    text_y = img_array.shape[0] - 10  # 10 pixels from the bottom

    # Put the text on the image
    cv2.putText(img_array, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.imshow(runtype_label, img_array)
    cv2.waitKey(0)

class GridEnvironment:
    def __init__(self, goal_idx=0, size=10):
        self.goals = [(1, 3), (size-1, 3), (3, 1), (3, size-1)]
        self.starting_states = [(1, 3), (size-1, 3), (3, 1), (3, size-1)]

        self.size = size
        self.state = None
        self.goal = self.goals[goal_idx]
        self.traffic_light = 1  # Traffic light signal: 1 (green) or 0 (red)
        self.invalid_state_actions = []
        self.start = self.starting_states[random.randint(0,3)]
        self.reset()

    
    def set_invalid_state_actions(self, invalid_state_actions):
        self.invalid_state_actions = invalid_state_actions

    def reset(self):
        """
        Reset to a random starting state that is not equal to the current goal.
        Also reset to a traffic light value (of 1)
        """
        self.start = self.starting_states[random.randint(0,3)]

        while self.goal == self.start:
            self.start = self.starting_states[random.randint(0,3)]
        
        self.traffic_light = 1
        self.state = self.start
        
        return self.state, self.traffic_light
    
    def step(self, action):
        x, y = self.state
        ox, oy = self.state
        # SENW
        if action == 1:   # south
            y = max(0, y - 1)
        elif action == 3: # north
            y = min(self.size - 1, y + 1)
        elif action == 4: # east
            x = max(0, x - 1)
        elif action == 2: # west
            x = min(self.size - 1, x + 1)
        
        self.state = (x, y)
        reward = -1  # Default reward for each step
        
        if [int(ox), int(oy), int(self.traffic_light), int(action)] in self.invalid_state_actions:
            reward -= 60

        if self.state == self.goal:
            reward = 100  # Reward for reaching the goal
            done = True
        else:
            done = False

        self.traffic_light = np.random.choice([0, 1])  # Randomly change traffic light signal
        
        return (self.state, self.traffic_light), reward, done
    
    def get_state_index(self, state, traffic_light):
        x, y = state
        return x * self.size + y + (self.size * self.size * traffic_light)

class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.dense = layers.Dense(action_size, activation='linear')

    def call(self, inputs):
        return self.dense(inputs)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.create_model()
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def create_model(self):
        return tf.keras.Sequential([
            layers.Dense(self.action_size, input_shape=(self.state_size,), kernel_initializer=tf.random_uniform_initializer(0, 0.01))
        ])

    @tf.function
    def train_step(self, state, Q_target):
        with tf.GradientTape() as tape:
            Q_out = self.model(state)
            loss = self.loss_function(Q_target, Q_out)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def predict(self, state):
        return self.model(state)

    def query_model(self, env, state, traffic_light):
        state_index = env.get_state_index(state, traffic_light)
        state_one_hot = tf.one_hot(state_index, self.state_size)
        state_one_hot = tf.expand_dims(state_one_hot, 0)
        q_values = self.predict(state_one_hot)
        action = np.argmax(q_values)
        return action, np.max(q_values)

def train(env, num_episodes=1500, max_steps_per_episode=100, gamma=0.99, alpha=0.01, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.95):

    state_size = env.size * env.size * 2  # Each state now includes traffic light signal
    action_size = 5  # up, down, left, right, stay still
    agent = DQNAgent(state_size, action_size, learning_rate=alpha)
    
    traj_store = []
    for episode in tqdm(range(num_episodes)):
        state, traffic_light = env.reset()
        total_reward = 0
        traj = []
        for step in range(max_steps_per_episode):
            state_index = env.get_state_index(state, traffic_light)
            state_one_hot = tf.one_hot(state_index, state_size)
            state_one_hot = tf.expand_dims(state_one_hot, 0)
            
            if np.random.rand() < epsilon:
                action = np.random.choice(action_size)
            else:
                q_values = agent.predict(state_one_hot)
                action = np.argmax(q_values)
            
            next_state, reward, done = env.step(action)
            next_state_index = env.get_state_index(next_state[0], next_state[1])
            next_state_one_hot = tf.one_hot(next_state_index, state_size)
            next_state_one_hot = tf.expand_dims(next_state_one_hot, 0)
            
            target = reward
            if not done:
                next_q_values = agent.predict(next_state_one_hot)
                target += gamma * np.max(next_q_values)
            
            q_values = agent.predict(state_one_hot).numpy()
            q_values[0][action] = target
            q_values = tf.convert_to_tensor(q_values)
            
            loss = agent.train_step(state_one_hot, q_values)
            
            state, traffic_light = next_state
            total_reward += reward
            
            traj.append((state[0], state[1], traffic_light, reward))
            if done:
                break
        
        traj_store.append(traj)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode%200 == 199:
            print(f"Episode: {episode + 1}, Total reward: {total_reward}, Epsilon: {epsilon:.2f}")

    return agent, traj_store

def gpt_test(env, agent, num_episodes=500, max_steps_per_episode=100):
    state_size  = env.size * env.size * 2          # one-hot length
    total_rewards = []
    traj_store    = []

    for _ in tqdm(range(num_episodes)):
        state, traffic_light = env.reset()         # starting state
        total_reward = 0
        traj = []                                  # will hold (x, y, tls, a)

        for _ in range(max_steps_per_episode):
            # --- choose action from CURRENT state ---------------------------
            idx          = env.get_state_index(state, traffic_light)
            s_one_hot    = tf.expand_dims(tf.one_hot(idx, state_size), 0)
            q_values     = agent.predict(s_one_hot)
            action       = int(np.argmax(q_values))  # ensure Python int

            # --- store (s, a) pair BEFORE taking the step ------------------
            traj.append((state[0], state[1], traffic_light, action))

            # --- environment transition ------------------------------------
            (next_x, next_y), reward, done = env.step(action)
            total_reward += reward

            # advance to next state
            state         = (next_x, next_y)
            traffic_light = env.traffic_light       # or however you read it

            if done:
                break

        traj_store.append(traj)
        total_rewards.append(total_reward)

    return total_rewards, traj_store

def test(env, agent, num_episodes=500, max_steps_per_episode=100):
    state_size = env.size * env.size * 2  # Each state now includes traffic light signal
    action_size = 5  # up, down, left, right, stay still
    total_rewards = []
    traj_store = []
    for episode in tqdm(range(num_episodes)):
        state, traffic_light = env.reset()
        total_reward = 0
        # traj = [(state[0], state[1], traffic_light, 0, 0)]
        traj = []
        for step in range(max_steps_per_episode):
            state_index = env.get_state_index(state, traffic_light)
            state_one_hot = tf.one_hot(state_index, state_size)
            state_one_hot = tf.expand_dims(state_one_hot, 0)
            
            q_values = agent.predict(state_one_hot)
            action = np.argmax(q_values)
            
            next_state, reward, done = env.step(action)
            next_state_index = env.get_state_index(next_state[0], next_state[1])
            next_state_one_hot = tf.one_hot(next_state_index, state_size)
            next_state_one_hot = tf.expand_dims(next_state_one_hot, 0)
            
            # Record current state and action take in it.
            traj.append((state[0], state[1], traffic_light, reward, action))

            state, traffic_light = next_state
            total_reward += reward
            
            # traj.append((state[0], state[1], traffic_light, reward, action))

            if done:
                break
        
        traj_store.append(traj)
        total_rewards.append(total_reward)

    return total_rewards, traj_store

def visualize_test_trajectories(test_trajectories, goal = (3,9), num=20, size=11):

    """
    Input: A list of trajectories generated by the agent
    Output: Step-by-step visualization of states in each trajectory T
    """
    for t in test_trajectories[:num]:
        for stp in t:
            plotx, ploty, tls, reward = (stp[0], stp[1], stp[2], stp[3])
            label = f"{plotx},{ploty},{str(stp[-1])}, {str(reward)}"
            original_paper_plot_state(plotx, ploty, tls, label, goal[0],goal[1], size=size)
            # input()

if __name__ == "__main__":

    invalid_state_actions = read_local_invalid_state_actions() 

    start = (9,3)
    goal = (3,9)
    
    env = GridEnvironment(start, goal)
    env.set_invalid_state_actions(invalid_state_actions)
    agent, traj_store = train(env, num_episodes=1500)

    total_rewards, test_trajectories = test(env, agent)
    result = {"Number Episodes":"1200", "Avg Reward":sum(total_rewards)/len(total_rewards)} 
    visualize_test_trajectories(test_trajectories, num=10)

