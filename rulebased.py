import matplotlib.pyplot as plt

import Sim
from agent import TTC


def draw_st_fig(st, sp):
    plt.plot(st, sp, label="ego")
    # for i in range(1, len(sp)):
    #     sps = sp[i]
    #     plt.plot(st[:len(sps)], sps, label="car{}".format(i))
    plt.xlabel("step")
    plt.ylabel("speed")
    plt.show()


def valid():
    env = Sim.make("t")
    ob = env.reset()
    agent = TTC(len(env.action_space))
    step_array = []
    sp_array = []
    while True:
        action = agent.get_action(ob)
        ob, reward, done, step = env.step(action)
        env.render()
        if done:
            draw_st_fig(step_array, sp_array)
            break
        step_array.append(step)
        sp_array.append(ob.ego.v)
    env.close()


if __name__ == "__main__":
    valid()
