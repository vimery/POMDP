import Sim
from agent import *
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
    env = Sim.make("full")
    observation = env.reset()
    agent = DQNAgent()

    speed_array = [[]]
    step_array = []

    max_run_times = 10
    cur_run_times = 0

    collision_times = 0
    success_times = 0
    time_out_times = 0

    while cur_run_times < max_run_times:
        action = agent.get_action(observation.get_array())
        observation, done, step = env.step(action)
        # env.render()

        if done:
            cur_run_times += 1
            observation = env.reset()
            if done == -1:
                collision_times += 1
            elif done == 1:
                success_times += 1
            else:
                time_out_times += 1
            step_array.append(step)
            # pg.quit()
            # draw_st_fig(step_array, speed_array)
            continue
            # break
        # step_array.append(step)
        # speed_array[0].append(observation.ego.state.v)
        # for i in range(1, len(observation.others) + 1):
        #     if i >= len(speed_array):
        #         speed_array.append([])
        #     speed_array[i].append(state.others[i - 1].state.v)

        # print("current step is: {}".format(step))
        # print("current position of vehicles are: ")
        # print("x: {}, y: {}, v: {}, action: {}".format(state.ego.state.x, state.ego.state.y, state.ego.state.v,
        #                                                state.ego.action))
        # for observation in state.others:
        #     print(
        #         "x: {}, y: {}, v: {}, action: {}".format(observation.state.x, observation.state.y, observation.state.v,
        #                                                  observation.a))

    print("collide rate: {}%".format(collision_times * 100 / max_run_times))
    print("success rate: {}%".format(success_times * 100 / max_run_times))
    print("time out rate: {}%".format(time_out_times * 100 / max_run_times))
    print("average pass steps: {}".format(np.average(step_array)))
    pg.quit()
