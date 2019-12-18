import matplotlib.pyplot as plt

import Sim
import os
from agent import *

MODEL_PATH = "data/DQN"


def draw_st_fig(st, sp):
    plt.plot(st, sp[0], label="ego")
    for i in range(1, len(sp)):
        steps = sp[i]
        plt.plot(st[:len(steps)], steps, label="car{}".format(i))
    plt.xlabel("step")
    plt.ylabel("speed")
    plt.show()


def plot_array(array, title, xlabel, ylabel):
    plt.figure()
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(array)
    plt.show()


def _get_ob(m_ob, ag):
    if not m_ob:
        return None
    x = torch.tensor(m_ob, dtype=torch.float)
    x = x.view((1, -1)).to(ag.device)
    return x


def learn():
    env = Sim.make("full")
    ob = env.reset().get_array()
    agent = DQNAgent(sum([len(l) for l in ob]), len(env.action_space))
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)

    step_array = []
    reward_array = []

    num_episodes = 1000

    step = 0

    max_reward = float("-inf")

    for i in range(num_episodes):
        cumulative_reward = 0
        while True:
            action = agent.get_action(_get_ob(ob, agent), step)
            new_ob, reward, done, step = env.step(action.item())
            new_ob = new_ob.get_array() if new_ob else None
            cumulative_reward += reward
            reward = torch.tensor([reward], device=agent.device, dtype=torch.long)

            agent.memory.push(_get_ob(ob, agent), action, _get_ob(new_ob, agent), reward)
            ob = new_ob

            agent.learn()
            # env.render()

            if done:
                ob = env.reset().get_array()
                step_array.append(step)
                reward_array.append(cumulative_reward)
                print(done, cumulative_reward)
                break
        # Update the target network, copying all weights and biases in DQN
        if i % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            mean_reward = np.mean(reward_array[i - 20:i])
            if mean_reward > max_reward:
                max_reward = mean_reward
                agent.save(MODEL_PATH)

    plot_array(step_array, "steps in Training", "episodes", "steps")
    plot_array(reward_array, "rewards in Training", "episodes", "reward")
    env.close()


def valid():
    env = Sim.make("full")
    ob = env.reset().get_array()
    agent = DQNAgent(sum([len(l) for l in ob]), len(env.action_space))
    agent.load(MODEL_PATH)

    num_episodes = 100
    collide_times = 0
    timeout_times = 0
    reward_array = []
    step_array = []
    step = 0
    for i in range(num_episodes):
        cumulative_reward = 0
        while True:
            action = agent.get_action(_get_ob(ob, agent), step)
            ob, reward, done, step = env.step(action.item())
            ob = ob.get_array() if ob else None
            cumulative_reward += reward

            # env.render()
            if done:
                collide_times += done == -1
                timeout_times += done == -2
                reward_array.append(cumulative_reward)
                step_array.append(step)
                break
    print("Total episodes in test: {}".format(num_episodes))
    print("collide rate: {}".format(collide_times / num_episodes))
    print("timeout rate: {}".format(timeout_times / num_episodes))
    print("success rate: {}".format((num_episodes - collide_times - timeout_times) / num_episodes))
    plot_array(reward_array, "cumulative reward in test", "episodes", "reward")
    plot_array(reward_array, "steps in test", "episodes", "reward")
    env.close()


if __name__ == '__main__':
    learn()
    valid()
