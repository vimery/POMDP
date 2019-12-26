import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
from agent import DQNAgent
import torch
from learnRL import plot_array

# 超参数
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # 最优选择动作百分比
GAMMA = 0.9  # 奖励递减参数
TARGET_REPLACE_ITER = 100  # Q 现实网络的更新频率
MEMORY_CAPACITY = 2000  # 记忆库大小
env = gym.make('CartPole-v0')  # 立杆子游戏
env = env.unwrapped

dqn = DQNAgent(env.observation_space.shape[0], env.action_space.n)
r_a = []
for i_episode in range(400):
    s = torch.tensor([env.reset().data], dtype=torch.float)
    r_s = 0
    while True:
        # env.render()  # 显示实验动画
        a = dqn.get_action(s)

        # 选动作, 得到环境反馈
        s_, _, done, info = env.step(a.item())

        # 修改 reward, 使 DDQNP 快速学习
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r_s += r1 + r2
        r = torch.tensor([r1 + r2], dtype=torch.float)

        s_ = torch.tensor([s_.data], dtype=torch.float)
        # 存记忆
        dqn.memory.push(s, a, s_, r)

        dqn.learn()

        if done:  # 如果回合结束, 进入下回合
            if len(dqn.memory) == dqn.memory_capacity:
                print('Ep: ', i_episode, '| Ep_r: ', round(r_s, 2))
                r_a.append(round(r_s, 2))
            break

        s = s_
plot_array(r_a, "reward", "EP", "reward")
