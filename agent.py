from tools import *


class Agent:
    """
    Agent: basic class represents a planner
    """

    def __init__(self):
        pass

    def get_action(self, state):
        pass


class TTC(Agent):
    """
    Time to collision, a model that describes how a driver will do with assumption of constant speed
    """

    def __init__(self):
        super().__init__()
        self.pre = 4
        self.ego = None

    def collide_predict(self, observation):
        ego = observation.ego
        for i in np.arange(0, self.pre, 0.1):
            ego_state = forward(ego.state, ego.route, i, ego.get_max_speed(), 0)
            if not ego_state[0]:
                continue
            for other in observation.others:
                other_state = forward(other.state, other.route, i, other.max_speed, 0)
                if other_state[0] and collide_detection(ego_state[0], ego_state[1], other_state[0], other_state[1],
                                                        ego.radius, other.radius):
                    return True
        return False

    def get_action(self, observation):
        if self.collide_predict(observation):
            return -5
        else:
            return 2


class WaitAndRun(Agent):

    def __init__(self):
        super().__init__()
        self.dis = 5

    def get_action(self, observation):
        action = 10
        for i in range(1, self.dis):
            new_state = self.ego.forward(distance=i)
            if new_state and new_state.collide(observation):
                action = -5
                break
        return action


class Constant(Agent):

    def __init__(self):
        super().__init__()
        self.acc = 0

    def get_action(self, state):
        return self.acc
