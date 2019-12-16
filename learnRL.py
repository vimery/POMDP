import Sim
from agent import *
import matplotlib.pyplot as plt
import numpy as np


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
    ob = env.reset()
    agent = DQNAgent()

    speed_array = [[]]
    step_array = []

    num_episodes = 10
    cur_run_times = 0

    collision_times = 0
    success_times = 0
    time_out_times = 0

    step = 0

    def _get_ob(m_ob):
        if not m_ob:
            return None
        x = torch.tensor(m_ob.get_array(), dtype=torch.float)
        x = x.view((1, -1)).to(agent.device)
        return x

    for i in range(num_episodes):
        while True:
            action = agent.get_action(_get_ob(ob), step, len(env.action_space))
            new_ob, reward, done, step = env.step(action.item())
            reward = torch.tensor([reward], device=agent.device, dtype=torch.long)

            agent.memory.push(_get_ob(ob), action, _get_ob(new_ob), reward)
            ob = new_ob

            agent.learn()
            env.render()

            if done:
                cur_run_times += 1
                ob = env.reset()
                if done == -1:
                    collision_times += 1
                elif done == 1:
                    success_times += 1
                else:
                    time_out_times += 1
                step_array.append(step)
                # pg.quit()
                # draw_st_fig(step_array, speed_array)
                break
        # Update the target network, copying all weights and biases in DQN
        if i % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        # step_array.append(step)
        # speed_array[0].append(observation.ego.v)
        # for i in range(1, len(observation.others) + 1):
        #     if i >= len(speed_array):
        #         speed_array.append([])
        #     speed_array[i].append(state.others[i - 1].state.v)

        # print("current step is: {}".format(step))
        # print("current position of vehicles are: ")
        # print("x: {}, y: {}, v: {}, action: {}".format(state.ego.x, state.ego.y, state.ego.v,
        #                                                state.ego.action))
        # for observation in state.others:
        #     print(
        #         "x: {}, y: {}, v: {}, action: {}".format(observation.x, observation.y, observation.v,
        #                                                  observation.a))

    print("collide rate: {}%".format(collision_times * 100 / num_episodes))
    print("success rate: {}%".format(success_times * 100 / num_episodes))
    print("time out rate: {}%".format(time_out_times * 100 / num_episodes))
    print("average pass steps: {}".format(np.average(step_array)))
    env.close()
