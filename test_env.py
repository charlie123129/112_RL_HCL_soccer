import soccer_twos
from soccer_twos.side_channels import EnvConfigurationChannel

env_channel = EnvConfigurationChannel()
env_channel.set_parameters(
    ball_state={
        "position": [14, -3],
        "velocity": [-1.2, 3],
    },
    players_states={
        3: {
            "position": [-5, 10],
            "rotation_y": 45,
            "velocity": [5, 0],
        }
    }
)
env = soccer_twos.make(
    render=True,
    flatten_branched=True,  # converts MultiDiscrete into Discrete action space
    variation=soccer_twos.EnvType.team_vs_policy,
    single_player=True,  # controls a single player while the other stays still
    opponent_policy=lambda *_: 0,  # opponents stay still
    env_channel=env_channel,
)

print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space)

team0_reward = 0
env.reset()

while True:
    continue

'''
while True:
    obs, reward, done, info = env.step(env.action_space.sample())
    team0_reward += reward
    if done:  # if any agent is done
        print("Total Reward: ", team0_reward)
        team0_reward = 0
        env.reset()
'''