import matplotlib.pyplot as plt

import Sim
from agent import *
import logging

MODEL_PATH = "data/DDQNP"
PIC_PATH = "result/"


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
    plt.savefig(PIC_PATH + title + ".png")


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
    # if os.path.exists(MODEL_PATH):
    #     agent.load(MODEL_PATH)

    reward_array = []
    average_reward = []

    num_episodes = 3000
    cur_ep = 0

    for i in range(num_episodes):
        cumulative_reward = 0
        while True:
            action = agent.get_action(_get_ob(ob, agent))
            new_ob, reward, done, step = env.step(action.item())
            new_ob = new_ob.get_array() if new_ob else None
            cumulative_reward += reward
            reward = torch.tensor([reward], device=agent.device, dtype=torch.long)

            agent.memory.push(_get_ob(ob, agent), action, _get_ob(new_ob, agent), reward)
            ob = new_ob

            agent.learn()
            # env.render()

            if done:
                cur_ep += 1
                ob = env.reset().get_array()
                reward_array.append(cumulative_reward)
                logging.info("EP:{} done:{} r:{}".format(cur_ep, done, cumulative_reward))
                if cur_ep % agent.target_update == 0:
                    mean = np.mean(reward_array[cur_ep - agent.target_update])
                    average_reward.append(mean)
                break

    plot_array(reward_array, "rewards in Training", "episodes", "reward")
    plot_array(average_reward, "average rewards in Training", "episodes*{}".format(agent.target_update), "reward")
    agent.save(MODEL_PATH)
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
    # step_array = []
    speed_array = []
    for i in range(num_episodes):
        cumulative_reward = 0
        while True:
            action = agent.get_action_without_exploration(_get_ob(ob, agent))
            ob, reward, done, step = env.step(action.item())
            # speed_array.append(ob.ego.v if ob else None)
            ob = ob.get_array() if ob else None
            cumulative_reward += reward

            # env.render()
            if done:
                ob = env.reset().get_array()
                collide_times += done == -1
                timeout_times += done == -2
                reward_array.append(cumulative_reward)
                # step_array.append(step)
                logging.info("done:{}, reward:{}".format(done, cumulative_reward))
                break
    logging.info("Total episodes in test: {}".format(num_episodes))
    logging.info("collide rate: {}".format(collide_times / num_episodes))
    logging.info("timeout rate: {}".format(timeout_times / num_episodes))
    logging.info("success rate: {}".format((num_episodes - collide_times - timeout_times) / num_episodes))
    plot_array(reward_array, "cumulative reward in test", "episodes", "reward")
    # plot_array(step_array, "steps in test", "episodes", "reward")
    # plot_array(speed_array, "speed variation", "step", "speed")
    env.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename="result/report.log", filemode="w")
    logging.info("===================== training begins ============================")
    learn()
    logging.info("===================== testing begins  ============================")
    valid()
