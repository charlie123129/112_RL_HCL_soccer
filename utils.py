import numpy as np


def compute_reward(observation, info):
    reward = 0
    # 假设每个时间点的观测线的数量
    num_rays = 42

    for i in [1,2,3,4,5,6,7,8,9,10,11,34,35,36]:
        # 提取当前观测线的数据
        ray_data = observation[i * 8 : (i + 1) * 8]

        # 进球獎勵
        # if ray_data[2] == 1:  # 'opposing_goal' is observed
        #    reward += 10.0 * (1 - ray_data[7])  # 进球越近，獎勵越大

        # 防守獎勵
        # if ray_data[1] == 1:  # 'our_goal' is observed
        #    reward -= 5.0 * (1 - ray_data[7])  # 防守越远，獎勵越大

        # 控球獎勵
        if ray_data[0] == 1:  # 'ball' is observed
            reward += 1.0 * (30 - ray_data[7])  # 控球越近，獎勵越大

        # 避免碰撞獎勵
        # if ray_data[3] == 1 or ray_data[4] == 1 or ray_data[5] == 1:  # 'wall', 'teammate', or 'opponent'
        #    reward -= 0.5  # 避免不必要的碰撞

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
