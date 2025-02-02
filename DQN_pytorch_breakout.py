import random
import numpy as np
import gymnasium as gym
import cv2  # for image preprocessing
from collections import deque, namedtuple
import ale_py
import matplotlib.pyplot as plt  # For plotting

import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ["SDL_AUDIODRIVER"] = "dummy"  # Disable SDL audio

# Check device
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 1000  # update target network every 1000 steps
MEMORY_SIZE = 100000
NUM_EPISODES = 50000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000000  # decay over 1M frames
TRAIN_START = 50000  # start training after 50k frames
MAX_FRAMES = 50000000  # total frames to train

# Preprocessing parameters
INPUT_WIDTH = 84
INPUT_HEIGHT = 84

# A simple replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# The DQN network architecture – adapted for RGB input.
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # output: (32, ~20, ~20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # output: (64, ~9, ~9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # output: (64, ~7, ~7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        return self.net(x)

# Preprocessing function: resize and normalize the RGB frame.
def preprocess_frame(frame):
    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    return frame

def main():
    gym.register_envs(ale_py)
    # Create the environment using Gymnasium.
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env.unwrapped.ale.setBool('sound', False)
    env = gym.wrappers.RecordVideo(
        env,
        episode_trigger=lambda episode: episode % 200 == 0,
        video_folder="saved-video-folder",
        name_prefix="video-",
    )
    
    num_actions = env.action_space.n
    print(f"Number of actions: {num_actions}")
    
    # Initialize networks
    policy_net = DQN(num_actions).to(device)
    target_net = DQN(num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0
    episode_rewards = []
    frame_count = 0

    # Set up interactive plotting.
    plt.ion()  # turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 6))
    reward_line, = ax.plot([], [], label="Reward per Episode")
    avg_line, = ax.plot([], [], label="Running Average (window=50)", color="orange")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Performance")
    ax.legend()
    ax.grid()
    
    window = 50  # window size for running average

    def update_plot():
        episodes = np.arange(1, len(episode_rewards) + 1)
        reward_line.set_data(episodes, episode_rewards)
        if len(episode_rewards) >= window:
            running_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            avg_line.set_data(np.arange(window, len(episode_rewards)+1), running_avg)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def select_action(state, eps_threshold):
        nonlocal steps_done
        if random.random() > eps_threshold:
            with torch.no_grad():
                q_values = policy_net(state.to(device))
                return q_values.max(1)[1].item()
        else:
            return random.randrange(num_actions)

    for i_episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        state = preprocess_frame(state)
        state = torch.from_numpy(state).unsqueeze(0)  # remain on CPU

        episode_reward = 0
        done = False

        while not done:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            action = select_action(state, eps_threshold)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            frame_count += 1
            steps_done += 1

            if not done:
                next_state = preprocess_frame(next_obs)
                next_state = torch.from_numpy(next_state).unsqueeze(0)
            else:
                next_state = None

            memory.push(state, action, reward, next_state, done)

            if next_state is not None:
                state = next_state

            if len(memory) > TRAIN_START and frame_count % 4 == 0:
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.tensor(
                    tuple(s is not None for s in batch.next_state), device=device, dtype=torch.bool
                )
                if any(non_final_mask):
                    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
                else:
                    non_final_next_states = torch.empty(0, device=device)
                
                state_batch = torch.cat(batch.state).to(device)
                action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward, device=device)

                state_action_values = policy_net(state_batch).gather(1, action_batch)

                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                if non_final_next_states.size(0) > 0:
                    with torch.no_grad():
                        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
                
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch

                loss = nn.functional.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)
                optimizer.step()

                if steps_done % TARGET_UPDATE_FREQ == 0:

                    target_net.load_state_dict(policy_net.state_dict())

            if frame_count >= MAX_FRAMES:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {i_episode} - Reward: {episode_reward} - Epsilon: {eps_threshold:.4f} - Total Steps: {steps_done}")
        update_plot()  # update the plot after each episode

        if frame_count >= MAX_FRAMES:
            print("Reached maximum training frames.")
            break

    env.close()
    torch.save(policy_net.state_dict(), "dqn_breakout_rgb.pth")
    print("Training complete and model saved.")

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()