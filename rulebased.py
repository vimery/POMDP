import matplotlib.pyplot as plt

import Sim
from agent import TTC


def valid():
    env = Sim.make("full")
    ob = env.reset()
    agent = TTC(len(env.action_space))
    while True:
        action = agent.get_action(ob)
        ob, reward, done, step = env.step(action)

        env.render()
        if done:
            break
    env.close()


if __name__ == "__main__":
    valid()
