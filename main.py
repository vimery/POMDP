from env import Env
from tools import InterParam
from agent import Constant

if __name__ == '__main__':
    params = InterParam()
    env = Env(params)
    state = env.reset()
    agent = Constant()

    while True:
        action = agent.get_action(state)
        new_state, done, step = env.step(action)
        env.render()
        if done:
            print(done)
            state = env.reset()
        print("current step is: {}".format(step))
        print("current position of vehicles are: ")
        for vehicle in new_state:
            print("x: {}, y: {}".format(vehicle.x, vehicle.y))
