from env import *
from tools import InterParam
from agent import TTC

if __name__ == '__main__':
    params = InterParam()
    env = InfiniteEnv(params)
    state = env.reset()

    while True:
        new_state, done, step = env.step()
        env.render()
        if done:
            print(done)
            state = env.reset()
