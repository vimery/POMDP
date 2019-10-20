from env import Env
from tools import InterParam
from agent import Constant

if __name__ == '__main__':
    params = InterParam()
    env = Env(params)
    state = env.reset()
    agent = Constant()

    step = 0
    while True:
        action = agent.get_action()
        new_state, done = env.step(action)
        env.render()
        if done:
            break
        print("current step is: {}".format(step))
        print("current position of vehicles are: ")
        for vehicle in new_state:
            print("x: {}, y: {}".format(vehicle.state.x, vehicle.state.y))
        step += 1
