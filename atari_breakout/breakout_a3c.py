import gymnasium as gym
import ale_py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import threading
from skimage.color import rgb2gray
from skimage.transform import resize

# Global variables
EPISODES = 8000000
env_name = "ALE/Breakout-v5"  # Using gymnasium environment name
# Check device
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Global episode counter
episode = 0

def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    return processed_observe

class ActorCriticNet(nn.Module):
    def __init__(self, action_size):
        super(ActorCriticNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 9 * 9, 256)
        self.policy = nn.Linear(256, action_size)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        policy = torch.softmax(self.policy(x), dim=1)
        value = self.value(x)
        return policy, value

class Agent(threading.Thread):
    def __init__(self, global_model, optimizer, action_size):
        threading.Thread.__init__(self)
        self.global_model = global_model
        self.optimizer = optimizer
        self.local_model = ActorCriticNet(action_size).to(device)
        self.local_model.load_state_dict(global_model.state_dict())
        self.action_size = action_size
        self.discount_factor = 0.99
        self.t_max = 20
        self.states, self.actions, self.rewards = [], [], []

    def run(self):
        global episode
        # Create the environment using gymnasium (render_mode can be adjusted as needed)
        gym.register_envs(ale_py)

        env = gym.make(env_name, render_mode="rgb_array")
        while episode < EPISODES:
            # Reset environment using gymnasium API
            observe, info = env.reset()
            next_observe = observe

            # Perform a random number of no-op steps to start the game
            no_ops = random.randint(1, 30)
            for _ in range(no_ops):
                observe, reward, terminated, truncated, info = env.step(1)
                if terminated or truncated:
                    observe, info = env.reset()
                next_observe = observe

            state = pre_processing(next_observe, observe)
            history = np.stack([state] * 4, axis=0) 
            history = np.expand_dims(history, axis=0)  # shape: (1, 4, 84, 84)
            history = torch.tensor(history, dtype=torch.float32).to(device)
            done, score, t = False, 0, 0

            while not done:
                t += 1
                policy, _ = self.local_model(history / 255.)
                action = np.random.choice(self.action_size, p=policy.cpu().detach().numpy()[0])
                real_action = [1, 2, 3][action]  # Map 0,1,2 -> 1,2,3
                next_observe, reward, terminated, truncated, info = env.step(real_action)
                done = terminated or truncated
                next_state = pre_processing(next_observe, observe)
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                next_history = torch.cat([next_state, history[:, :3, :, :]], dim=1)

                self.states.append(history)
                self.actions.append(action)
                self.rewards.append(np.clip(reward, -1., 1.))

                if t >= self.t_max or done:
                    self.train_model(done)
                    self.local_model.load_state_dict(self.global_model.state_dict())
                    t = 0

                history = next_history
                observe = next_observe
                score += reward

            episode += 1
            print(f"Episode {episode}, Score: {score}")
        env.close()

    def train_model(self, done):
        R = 0 if done else self.local_model(self.states[-1] / 255.)[1].item()
        discounted = []
        for r in reversed(self.rewards):
            R = r + self.discount_factor * R
            discounted.insert(0, R)

        states = torch.cat(self.states)
        actions = torch.tensor(self.actions).to(device)
        discounted = torch.tensor(discounted, dtype=torch.float32).to(device)        
        policies, values = self.local_model(states / 255.)

        values = values.squeeze()
        advantages = discounted - values.detach()

        log_probs = torch.log(torch.gather(policies, 1, actions.unsqueeze(1)).squeeze() + 1e-10)
        actor_loss = -torch.sum(log_probs * advantages)
        critic_loss = nn.functional.mse_loss(values, discounted)
        entropy = -torch.sum(policies * torch.log(policies + 1e-10))

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        self.optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
            if local_param.grad is not None:
                global_param._grad = local_param.grad
       
        self.optimizer.step()

        self.states, self.actions, self.rewards = [], [], []

if __name__ == "__main__":
    global_model = ActorCriticNet(action_size=3).to(device)
    global_model.share_memory()
    optimizer = optim.RMSprop(global_model.parameters(), lr=2.5e-4, eps=0.01, alpha=0.99)
    threads = []
    for _ in range(8):
        agent = Agent(global_model, optimizer, action_size=3)
        agent.start()
        threads.append(agent)
    for thread in threads:
        thread.join()