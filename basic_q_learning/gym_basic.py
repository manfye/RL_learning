import gymnasium as gym
import matplotlib.pyplot as plt
import time

import ale_py
gym.register_envs(ale_py)

# env = gym.make('ALE/Breakout-v5')
# obs, info = env.reset()
# obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# env.close()

# env = gym.make("LunarLander-v3", render_mode="human")
env = gym.make("MountainCar-v0", render_mode="human", goal_velocity=0.1)  # default goal_velocity=0

print(env.action_space.sample())
print(env.observation_space)
# Reset the environment to generate the first observation




observation, info = env.reset(seed=42)


def simple_agent(observation):
    position, velocity = observation
    if -0.1 < position < 0.4:
        action = 2
    elif velocity < 0 and position < -0.2:
        action = 0
    else:
        action = 1
    return action


for steps in range(600):
#     # this is where you would insert your policy
    # action = env.action_space.sample()
    action = simple_agent(observation)
#     # step (transition) through the environment with the action
#     # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    print("Observation: ", observation)

#     # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
