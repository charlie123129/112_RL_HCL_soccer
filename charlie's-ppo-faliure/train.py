import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import soccer_twos
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt


# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 检测是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}.")

# 定义策略网络（Actor）
class Actor(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_size, 32)  # Adjusted to 128 units
        self.fc2 = nn.Linear(32, 27)       # Adjusted input to 128, output to 27
        self.fc3 = nn.Linear(27, action_size)  # Added fc3 layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)  # Apply softmax on the output of fc3
        return x


# 定义价值网络（Critic）
class Critic(nn.Module):
    def __init__(self, obs_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_size, 32)  # Adjusted to 128 units
        self.fc2 = nn.Linear(32, 27)       # Adjusted input to 128, output to 27
        self.fc3 = nn.Linear(27, 1)         # Added fc3 layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output of fc3
        return x



# Define PPO hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ACTOR_LR = 0.0001  # 降低Actor的学习率
CRITIC_LR = 0.0005  # 显著降低Critic的学习率
NUM_EPOCHS = 2500
STEPS_PER_EPOCH = 512

# 初始化环境
try:
    env = soccer_twos.make(
        render=True,
        flatten_branched=True,
        variation=soccer_twos.EnvType.team_vs_policy,
        single_player=True,
        opponent_policy=lambda *_: 0,
    )
except Exception as e:
    logging.error(f"Error initializing environment: {e}")
    raise

obs_size = env.observation_space.shape[0]
action_size = env.action_space.n

actor = Actor(obs_size, action_size).to(device)
critic = Critic(obs_size).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

def calculate_reward(observation, info):
    reward = 0
    # 假设每个时间点的观测线的数量
    num_rays = 42

    for i in [1,2,3,4,5,6,7,8,9,10,11,34,35,36]:
        # 提取当前观测线的数据
        ray_data = observation[i * 8 : (i + 1) * 8]

        # 进球獎勵
        if ray_data[2] == 1:  # 'opposing_goal' is observed
           reward += 10.0 * (1 - ray_data[7])  # 进球越近，獎勵越大

        # 防守獎勵
        #if ray_data[1] == 1:  # 'our_goal' is observed
        #   reward -= 5.0 * (1 - ray_data[7])  # 防守越远，獎勵越大

        # 控球獎勵
        #if ray_data[0] == 1:  # 'ball' is observed
        #    reward += 1.0 * (30 - ray_data[7])  # 控球越近，獎勵越大

        # 避免碰撞獎勵
        #if ray_data[3] == 1 or ray_data[4] == 1 or ray_data[5] == 1:  # 'wall', 'teammate', or 'opponent'
        #   reward -= 0.5  # 避免不必要的碰撞

    # 球門位置our:x<-17, opponent:x>14
    #        -3<y<3

    ball_position = info["ball_info"]["position"]
    distance_to_goal = np.linalg.norm(np.array([14, 0]) - ball_position)  # 对 x 和 y 轴坐标同时考虑
    reward += 1 / (distance_to_goal + 0.1)  # 距离越近，奖励越大

    # 对于 y 轴位置，只有当球在球门范围内时给予奖励
    if -3 <= ball_position[1] <= 3:
        reward += 1
    else:
        reward -= 1

    # 时间懲罰
    reward -= 0.1

    return reward

# 收集数据的函数
def collect_data(env, actor, steps):
    states, actions, rewards, dones, next_states, log_probs = [], [], [], [], [], []
    state = env.reset()

    for _ in range(steps):
        state_tensor = torch.from_numpy(state).float().to(device)
        action_probs = actor(state_tensor)
        distribution = Categorical(action_probs)
        action = distribution.sample()

        next_state,_, done, info = env.step(action.item())
        reward = calculate_reward(state, info)

        
        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        next_states.append(next_state)
        log_probs.append(distribution.log_prob(action).item())

        state = next_state if not done else env.reset()

    return states, actions, rewards, dones, next_states, log_probs

# 计算回报的函数
def compute_returns(rewards, dones, gamma):
    returns = []
    R = 0
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * (1 - dones[step])
        returns.insert(0, R)
    return returns

# 计算优势的函数
def compute_advantages(rewards, dones, values, next_values):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + GAMMA * next_values[i] * (1 - dones[i]) - values[i]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return advantages

# 更新网络的函数
def update_network(states, actions, advantages, old_log_probs, actor, critic, actor_optimizer, critic_optimizer, returns):
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device).squeeze()

    # Update actor
    actor_optimizer.zero_grad()
    new_probs = actor(states)
    new_distribution = Categorical(new_probs)
    new_log_probs = new_distribution.log_prob(actions)
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    actor_loss.backward()
    # Apply gradient clipping to actor
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=2.0)
    actor_optimizer.step()

    # Update critic
    critic_optimizer.zero_grad()
    new_values = critic(states).squeeze().to(device)
    critic_loss = nn.MSELoss()(new_values, returns)
    critic_loss.backward()
    # Apply gradient clipping to critic
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=2.0)
    critic_optimizer.step()

    return actor_loss.item(), critic_loss.item()

def evaluate_performance(env, actor, num_trials=10):

    rewards = []
    for _ in range(num_trials):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.from_numpy(state).float().to(device)
            action_probs = actor(state_tensor)
            action = torch.argmax(action_probs).item()
            state, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)

def save_model(model, path, filename):
    torch.save(model.state_dict(), os.path.join(path, filename))

def load_model(model, path, filename):
   model.load_state_dict(torch.load(os.path.join(path, filename), map_location=torch.device('cpu')))


# 主训练循环
model_path = 'train_saved_models'  # 模型保存路径
os.makedirs(model_path, exist_ok=True)  # 创建模型保存目录

reward_history = []  # 用于记录奖励历史
actor_losses, critic_losses = [], []

# 主训练循环
best_avg_reward = float('-inf')  # 初始化最高平均奖励
best_performance = float('-inf')  # 初始化最佳性能

for epoch in tqdm(range(NUM_EPOCHS), desc='Training Epochs'):
    
    states, actions, rewards, dones, next_states, old_log_probs = collect_data(env, actor, STEPS_PER_EPOCH)

    # 计算回报和优势
    values = critic(torch.tensor(np.array(states), dtype=torch.float32).to(device)).squeeze()
    next_values = critic(torch.tensor(np.array(next_states), dtype=torch.float32).to(device)).squeeze()
    returns = compute_returns(rewards, dones, GAMMA)
    advantages = compute_advantages(rewards, dones, values.detach().cpu().numpy(), next_values.detach().cpu().numpy())

    # 更新网络
    actor_loss, critic_loss = update_network(states, actions, advantages, old_log_probs, actor, critic, actor_optimizer, critic_optimizer, returns)
    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)

    # 计算平均奖励
    avg_reward = np.mean(rewards)
    reward_history.append(avg_reward)

    # 打印当前epoch的信息
    print(f"\nEpisode: {epoch}, Reward: {avg_reward}, "
          f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

    # 检查是否有最高的平均奖励
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        save_model(actor, model_path, 'best_avg_reward_actor.pt')
        save_model(critic, model_path, 'best_avg_reward_critic.pt')
        print(f"Epoch {epoch}: Best Average Reward Model Saved: {best_avg_reward}")


    # 性能评估
    current_performance = evaluate_performance(env, actor)
    if current_performance > best_performance:
        best_performance = current_performance
        save_model(actor, model_path, 'best_performance_actor.pt')
        save_model(critic, model_path, 'best_performance_critic.pt')
        print(f"Epoch {epoch}: Best Performance Model Saved: {best_performance}")        


    # 每10个epoch保存一次模型    
    if epoch % 10 == 0:
        
        save_model(actor, model_path, f'actor_epoch_{epoch}.pt')
        save_model(critic, model_path, f'critic_epoch_{epoch}.pt')
        print(f"Epoch {epoch}: Model Saved at Epoch {epoch}")

        plt.figure(figsize=(10, 5))
        plt.plot(reward_history)
        plt.xlabel('Epoch')
        plt.ylabel('Average Reward')
        plt.title('Reward History Over Epochs')
        plt.grid(True)
        plt.savefig(f'reward_history_epoch_{epoch}.png')

        # 绘制损失历史图表
        plt.figure(figsize=(10, 5))
        plt.plot(actor_losses, label='Actor Loss')
        plt.plot(critic_losses, label='Critic Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Actor & Critic Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'loss_history_epoch_{epoch}.png')


env.close()