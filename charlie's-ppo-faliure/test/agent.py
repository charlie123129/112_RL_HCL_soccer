import os

from gym_unity.envs import ActionFlattener
import numpy as np
import torch
from soccer_twos import AgentInterface

from .model import QNetwork


class TeamAgent(AgentInterface):
    """
    An agent definition for policies trained with DQN on `team_vs_policy` variation with `single_player=True`.
    """

    def __init__(self, env):
        self.name = "charlie"
        # use flattened, Discrete actions instead of default MultiDiscrete
        self.flattener = ActionFlattener(env.action_space.nvec)
        # this agent's model works with team_vs_policy variation of the env
        # so we need to convert observations & actions
        self.model = QNetwork(
            env.observation_space.shape[0],
            self.flattener.action_space.n,
            seed=0,
        )
        # check if weights exist, load weights & put model in eval mode
        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "best_actor.pth")
        if os.path.isfile(weights_path):
            self.model.load_state_dict(torch.load(weights_path))
        else:
            print("Checkpoint not found.")
        self.model.eval()

    def act(self, observation):
        actions = {}
        for player_id in observation:
            # 将观测转换为张量
            state = torch.from_numpy(observation[player_id]).float().unsqueeze(0)

            # 预测每个动作的Q值
            with torch.no_grad():
                action_values = self.model(state)

            # 选择具有最高Q值的动作
            action = np.argmax(action_values.cpu().data.numpy())

            # 将选择的动作转换为环境可接受的格式
            actions[player_id] = self.flattener.lookup_action(action)

        return actions