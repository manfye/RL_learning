from collections import deque
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import ale_py
from collections import deque
import random

import numpy as np

import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses warnings and info
tf.get_logger().setLevel('ERROR')  # Suppresses additional TensorFlow logs

gym.register_envs(ale_py)

env = gym.make("CartPole-v1", render_mode="human")

env.reset()


num_observation_space = env.observation_space.shape[0]
print("num_observation_space: ", num_observation_space)

num_action_space = env.action_space.n
print("num_action_space: ", num_action_space)

# input shape = 4 

model = models.Sequential()
model.add(layers.Dense(16, input_shape=(num_observation_space,), activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(num_action_space, activation='linear'))


# This yields 690 parameters
# 4 observations * 16 {neurons} + 16 {bias} + (16*32) + 32 + (32*2)+2 = 690
# 80 first layer + 544 second layer + 66 third layer = 690
model.summary()


target_model = models.clone_model(model)

EPOCHS = 1000
epsilon = 1.0
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
GAMMA = 0.95

def epsilon_greedy_action_selection(model, epsilon, observation):
    if np.random.rand() > epsilon:
        prediction = model.predict(observation)
        action = np.argmax(prediction)

    else:
        action = np.random.randint(0, num_action_space)
    return action


replay_buffer = deque(maxlen=20000)
update_target_model = 10        

def replay(replay_buffer, batch_size, model, target_model):
    
    # As long as the buffer has not enough elements we do nothing
    if len(replay_buffer) < batch_size: 
        return
    
    # Take a random sample from the buffer with size batch_size
    samples = random.sample(replay_buffer, batch_size)  
    
    # to store the targets predicted by the target network for training
    target_batch = []  
    
    # Efficient way to handle the sample by using the zip functionality
    zipped_samples = list(zip(*samples))  
    states, actions, rewards, new_states, dones = zipped_samples  # S,A,R,S',D
    
    states_array = np.array(states).squeeze(axis=1)  # converts shape (batch_size, 1, 4) -> (batch_size, 4)
    new_states_array = np.array(new_states).squeeze(axis=1)
    
    # Predict targets for all states from the sample using the target model
    targets = target_model.predict(states_array)
    
    # Predict Q-Values for all new states from the sample using the main model
    q_values = model.predict(new_states_array)  
    
    # Now we loop over all predicted values to compute the actual targets
    for i in range(batch_size):  
        
        # Take the maximum Q-Value for each sample
        # print(q_values)
        q_value = np.max(q_values[i])        
        
        # Store the ith target in order to update it according to the formula
        target = targets[i].copy()  
        if dones[i]:
            target[actions[i]] = rewards[i]
        else:
            target[actions[i]] = rewards[i] + q_value * GAMMA
        target_batch.append(target)

    # Fit the model based on the states and the updated targets for 1 epoch
    model.fit(states_array, np.array(target_batch), epochs=1, verbose=0)

def update_model_handler(epoch, update_target_model, model, target_model):
    if epoch > 0 and epoch % update_target_model == 0:
        # not updating the target model always, but only every 10 epochs
        target_model.set_weights(model.get_weights())


model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')

scores = []
avg_scores = []
moving_avg_window = 25

# Set up interactive plotting
plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label="Episode Score")
line2, = ax.plot([], [], label="Running Average")
ax.set_xlabel("Episode")
ax.set_ylabel("Score")
ax.legend()

best_so_far = 0
for epoch in range(EPOCHS):
    observation, info = env.reset()    
    # Keras expects the input to be of shape [1, X] thus we have to reshape
    observation = observation.reshape([1, 4])
    done = False  
    
    points = 0
    while not done:  # as long current run is active
        
        # Select action acc. to strategy
        action = epsilon_greedy_action_selection(model, epsilon, observation)
        
        # Perform action and get next state
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_observation = next_observation.reshape([1, 4])  # Reshape!!
        replay_buffer.append((observation, action, reward, next_observation, done))  # Update the replay buffer
        observation = next_observation  # update the observation
        points+=1

        # Most important step! Training the model by replaying
        replay(replay_buffer, 32, model, target_model)

    
    epsilon *= EPSILON_DECAY  # Reduce epsilon
    
    # Check if we need to update the target model
    update_model_handler(epoch, update_target_model, model, target_model)
    scores.append(points)

    # Compute running average over the last moving_avg_window episodes
    if len(scores) < moving_avg_window:
        avg_scores.append(np.mean(scores))
    else:
        avg_scores.append(np.mean(scores[-moving_avg_window:]))

    if points > best_so_far:
        best_so_far = points
    if epoch %25 == 0:
        print(f"{epoch}: Points reached: {points} - epsilon: {epsilon} - Best: {best_so_far}")

 # Update plot every episode
    line1.set_xdata(np.arange(len(scores)))
    line1.set_ydata(scores)
    line2.set_xdata(np.arange(len(avg_scores)))
    line2.set_ydata(avg_scores)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)

# Finalize the plot
plt.ioff()
plt.show()