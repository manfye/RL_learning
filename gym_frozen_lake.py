import gymnasium as gym
import matplotlib.pyplot as plt
import time
import numpy as np
import ale_py
gym.register_envs(ale_py)

# env = gym.make('ALE/Breakout-v5')
# obs, info = env.reset()
# obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# env.close()

# env = gym.make("LunarLander-v3", render_mode="human")
# env = gym.make("MountainCar-v0", render_mode="human", goal_velocity=0.1)  # default goal_velocity=0

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1',
                desc=None, map_name="4x4",
                is_slippery=False,
                max_episode_steps=100,
                render_mode="human"  # Change to "human" if you want visualization
                )

# Q-Learning parameters
EPOCHS = 20000  # Number of games to play
ALPHA = 0.8  # Learning rate
GAMMA = 0.99  # Discount factor
EPSILON = 1.0  # Exploration rate
MAX_EPSILON = 1.0  # Initial exploration probability
MIN_EPSILON = 0.01  # Minimum exploration probability
DECAY_RATE = 0.001  # Exponential decay rate for exploration prob

# Initialize Q-table
action_size = env.action_space.n  # Number of actions (4: left, down, right, up)
state_size = env.observation_space.n  # Number of states (16 for 4x4 FrozenLake)
q_table = np.zeros([state_size, action_size])

# Function to choose action using epsilon-greedy strategy
def epsilon_greedy_action_selection(epsilon, q_table, state):
    if np.random.random() > epsilon:
        # Choose the action with the highest Q-value
        action = np.argmax(q_table[state, :])
    else:
        # Choose a random action
        action = env.action_space.sample()
    return action

# Function to update Q-value
def compute_next_q_value(old_q_value, reward, next_optimal_q_value):
    next_q_value = old_q_value + ALPHA * (reward + GAMMA * next_optimal_q_value - old_q_value)
    return next_q_value

# Function to reduce epsilon
def reduce_epsilon(epsilon, epoch):
    # print("reducing epsilon")
    return MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * epoch)

def save_q_table(q_table, steps_trained):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    np.savez("q_table.npz", q_table=q_table, steps=steps_trained, timestamp=timestamp)
    print(f"✅ Q-Table saved successfully! Steps trained: {steps_trained}, Time: {timestamp}")

import numpy as np
import datetime

# Training loop
epsilon = EPSILON

log_interval = 100

################################################

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion() # Turn on interactive mode
fig.canvas.draw()
epoch_plot = []
total_rewards_plot = []
rewards = []

################################################
total_rewards = 0

for episode in range(EPOCHS):
    # Reset environment and get the initial state
    state, info = env.reset(seed=42)  # Ensure proper unpacking
    state = int(state)  # Ensure state is an integer for indexing
    done = False
    step_count = 0  # ✅ New variable to track steps per episode

    while not done:
        step_count += 1  # ✅ Increment step count for each move

        # Choose action
        action = epsilon_greedy_action_selection(epsilon, q_table, state)

        # Take action and observe new state and reward
        new_state, reward, terminated, truncated, info = env.step(action)
        new_state = int(new_state)  # Convert to integer for Q-table indexing
        
        # Get current Q-value
        old_q_value = q_table[state, action]

        # Get best Q-value for the next state
        next_optimal_q_value = np.max(q_table[new_state, :])

        # Update Q-table
        q_table[state, action] = compute_next_q_value(old_q_value, reward, next_optimal_q_value)



        # Accumulate rewards
        total_rewards += reward  # Track total reward per episode
        
        if reward == 1:
            print("Episode: ", episode, "Total Reward: ", total_rewards, "Epsilon: ", epsilon)
            print("Found the goal after ", step_count, " steps")
            print("Q-table: ", q_table)
            
        # Move to the new state
        state = new_state

        # Check if the episode is finished
        done = terminated or truncated

    # Reduce epsilon after each episode
    epsilon = reduce_epsilon(epsilon, episode)
    rewards.append(total_rewards)

    epoch_plot.append(episode)
    total_rewards_plot.append(total_rewards)

    ax.clear()
    ax.plot(epoch_plot, total_rewards_plot)
    fig.canvas.draw()
    plt.show()  

    
    if episode % log_interval == 0:
        print("Episode: ", episode, "Total Reward: ", total_rewards, "Epsilon: ", epsilon)
        save_q_table(q_table, episode)


env.close()