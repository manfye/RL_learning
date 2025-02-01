import gymnasium as gym
import matplotlib.pyplot as plt
import time
import numpy as np
import ale_py
import numpy as np
import datetime

gym.register_envs(ale_py)
# https://gymnasium.farama.org/environments/classic_control/cart_pole/


# Observation Space

# 0 - Cart Position -4.8 to 4.8

# 1 - Cart Velocity -Inf to Inf

# 2 - Pole Angle -0.418 rad (-24°) to 0.418 rad (24°)
                                               
# 3 - Pole Angular Velocity -Inf to Inf



# Create the FrozenLake environment
# env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.make("CartPole-v1", render_mode="human")

def create_bins(num_bins_per_action=10):
    bins_cart_position = np.linspace(-4.8, 4.8, num_bins_per_action)  # bins for the cart position
    bins_cart_velocity = np.linspace(-5, 5, num_bins_per_action)  # bins for the cart velocity
    bins_pole_angle = np.linspace(-0.418, 0.418, num_bins_per_action)  # bins for the pole angle
    bins_pole_angular_velocity = np.linspace(-5, 5, num_bins_per_action)  # bins for the pole angular velocity
    bins = np.array([bins_cart_position, bins_cart_velocity, bins_pole_angle, bins_pole_angular_velocity])  # merge them
    return bins




# Q-Learning parameters
EPOCHS = 20000
ALPHA = 0.8
GAMMA = 0.9

# Exploration vs. Exploitation parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.001             # Exponential decay rate for exploration prob

def discretize_observation(observations, bins):
    binned_observations = []
    for i, observation in enumerate(observations):
        discretized_observation = np.digitize(observation, bins[i])
        binned_observations.append(discretized_observation)
    return tuple(binned_observations) # Important for later indexing

# Function to choose action using epsilon-greedy strategy
def epsilon_greedy_action_selection(epsilon, q_table, discrete_state):
    '''
    Returns an action for the agent. Note how it uses a random number to decide on
    exploration versus explotation trade-off.
    '''
    random_number = np.random.random()
    
    # EXPLOITATION, USE BEST Q(s,a) Value
    if random_number > epsilon:

        action = np.argmax(q_table[discrete_state])

    # EXPLORATION, USE A RANDOM ACTION
    else:
        # Return a random 0,1,2,3 action
        action = np.random.randint(0, env.action_space.n)

    return action

# Function to update Q-value
def compute_next_q_value(old_q_value, reward, next_optimal_q_value):
    next_q_value = old_q_value + ALPHA * (reward + GAMMA * next_optimal_q_value - old_q_value)
    return next_q_value

BURN_IN = 1
epsilon = 1

EPSILON_END= 10000
EPSILON_REDUCE = 0.0001

def reduce_epsilon(epsilon, epoch):
    if BURN_IN <= epoch <= EPSILON_END:
        epsilon-= EPSILON_REDUCE
    return epsilon

def save_q_table(q_table, steps_trained):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    np.savez("q_table.npz", q_table=q_table, steps=steps_trained, timestamp=timestamp)
    print(f"✅ Q-Table saved successfully! Steps trained: {steps_trained}, Time: {timestamp}")

def fail(done, points, reward):
    if done and points < 150:
        reward = -200
    return reward


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


log_interval = 100  # How often do we update the plot? (Just for performance reasons)
render_interval = 2000  # How often to render the game during training (If you want to watch your model learning)
### Here we set up the routine for the live plotting of the achieved points ######
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
fig.canvas.draw()
#############################################

NUM_BINS = 10
BINS = create_bins(NUM_BINS)  # Create the bins used for the rest of the notebook

q_table_shape = (NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, env.action_space.n)
q_table = np.zeros(q_table_shape)
print(q_table.shape)

points_log = []  # to store all achieved points
mean_points_log = []  # to store a running mean of the last 30 results
epochs = []  # store the epoch for plotting

for epoch in range(EPOCHS):
    state, info = env.reset(seed=42)  # Ensure proper unpacking

    discretized_state = discretize_observation(state, BINS)  # map the observation to the bins
    
    done = False  # to stop current run when cartpole falls down   
    points = 0  # store result
    
    # Track Epochs for Plotting Visualization
    epochs.append(epoch)
    
    
    while not done:  # Perform current run as long as done is False (as long as the cartpole is up)

        action = epsilon_greedy_action_selection(epsilon, q_table, discretized_state)  # Epsilon-Greedy Action Selection 
        next_state, reward, done, _, info = env.step(action)

        reward = fail(done, points, reward)  # Check if reward or fail state


        next_state_discretized = discretize_observation(next_state, BINS)  # map the next observation to the bins

        old_q_value =  q_table[discretized_state + (action,)]  # get the old Q-Value from the Q-Table
        next_optimal_q_value = np.max(q_table[next_state_discretized])  # Get the next optimal Q-Value
        

        next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)  # Compute next Q-Value
        q_table[discretized_state + (action,)] = next_q  # Insert next Q-Value into the table

        discretized_state = next_state_discretized  # Update the old state
        points += 1

    epsilon = reduce_epsilon(epsilon, epoch)  # Reduce epsilon
    points_log.append(points)  # log overall achieved points for the current epoch
    running_mean = round(np.mean(points_log[-30:]), 2)  # Compute running mean points over the last 30 epochs
    mean_points_log.append(running_mean)  # and log it
    
    ################ Plot the points and running mean ##################
    if epoch % log_interval == 0:
        ax.clear()
        ax.scatter(epochs, points_log)
        ax.plot(epochs, points_log)
        ax.plot(epochs, mean_points_log, label=f"Running Mean: {running_mean}")
        plt.legend()
        fig.canvas.draw()
  ######################################################################

env.close()