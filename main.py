from env import *
from tools import InterParam
from agent import TTC
import matplotlib.pyplot as plt


def draw_st_fig(st, sp):
    plt.plot(st, sp[0], label="ego")
    for i in range(1, len(sp)):
        steps = sp[i]
        plt.plot(st[:len(steps)], steps, label="car{}".format(i))
    plt.xlabel("step")
    plt.ylabel("speed")
    plt.show()


if __name__ == '__main__':
    params = InterParam()
    env = Env(params)
    state = env.reset()
    agent = TTC()

    speed_array = [[]]
    step_array = []

    while True:
        action = agent.get_action(state)
        state, done, step = env.step(action)
        env.render()

        if done:
            print(done)
            pg.quit()
            break
        step_array.append(env.step_count)
        speed_array[0].append(state.ego.state.v)
        for i in range(1, len(state.others) + 1):
            if i >= len(speed_array):
                speed_array.append([])
            speed_array[i].append(state.others[i-1].state.v)

        print("current step is: {}".format(step))
        print("current position of vehicles are: ")
        print("x: {}, y: {}, action: {}".format(state.ego.state.x, state.ego.state.y, state.ego.action))
        for observation in state.others:
            print("x: {}, y: {}, action: {}".format(observation.state.x, observation.state.y, observation.a))

    draw_st_fig(step_array, speed_array)
