import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import soccer_twos
from itertools import count
from utils import compute_reward
import matplotlib.pyplot as plt


# 定義 DQN 網絡
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# 簡單的回放記憶
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN 代理
class DQNAgent:
    def __init__(self, input_dim, output_dim, learning_rate=1e-4):
        self.policy_net = DQN(input_dim, output_dim).float()
        self.target_net = DQN(input_dim, output_dim).float()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(10000)

    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                return (
                    self.policy_net(torch.from_numpy(state).float().unsqueeze(0))
                    .max(1)[1]
                    .view(1, 1)
                )
        else:
            return torch.tensor([[random.randrange(output_dim)]], dtype=torch.long)

    def optimize(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch[2])), dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [
                torch.from_numpy(s).float().unsqueeze(0)
                for s in batch[2]
                if s is not None
            ]
        )

        state_batch = torch.cat(
            [torch.from_numpy(s).float().unsqueeze(0) for s in batch[0]]
        )
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat([torch.tensor([r]) for r in batch[3]])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(batch_size)
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = nn.SmoothL1Loss()(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 初始化環境和代理
env = soccer_twos.make(
    render=False,
    flatten_branched=True,  # converts MultiDiscrete into Discrete action space
    variation=soccer_twos.EnvType.team_vs_policy,
    single_player=True,  # controls a single player while the other stays still
    opponent_policy=lambda *_: 0,  # opponents stay still
)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = DQNAgent(input_dim, output_dim)
episode_rewards = []

# 訓練參數
num_episodes = 3000
batch_size = 128
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 200

# 訓練循環
for episode in range(num_episodes):
    state = env.reset()
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
        -1.0 * episode / epsilon_decay
    )
    total_reward = 0

    for t in count():
        action = agent.select_action(state, epsilon)
        next_state, inreward, done, _info = env.step(action.item())
        reward = compute_reward(next_state, _info) + (10 * inreward)
        total_reward += reward
        reward = torch.tensor([reward])

        if done:
            next_state = None

        agent.memory.push(state, action, next_state, reward)

        state = next_state

        agent.optimize(batch_size, gamma)

        if done:
            episode_rewards.append(total_reward)
            print(f"Episode {episode}, Total Reward: {total_reward}")
            break

    if episode % 10 == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        save_path = "/home/m11107326/git-repo/Soccer-Environment/model/"
        filename = f"{save_path}dqn_policy_net_episode_{episode}.pt"
        torch.save(agent.policy_net.state_dict(), filename)
        print(f"Model saved as {filename}")


torch.save(agent.policy_net, f"{save_path}dqn_policy_net.pt")

# 绘制学习曲线
plt.plot(range(len(episode_rewards)), episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve")
plt.show()

env.close()
