from env import Env
import random


class Agent:

    @staticmethod
    def choose_action(actions):
        return random.choice(actions)


if __name__ == '__main__':
    env = Env()
    env.reset()
    while True:
        new_state, reward, done = env.step(Agent.choose_action(env.actions))
        if done:
            break
