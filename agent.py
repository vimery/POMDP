class Agent:
    """
    Agent: basic class represents a planner
    """
    def __init__(self):
        pass

    def get_action(self):
        pass


class TTC(Agent):
    """
    Time to collision, a model that describes how a driver will do with assumption of constant speed
    """

    def __init__(self, observations):
        super(TTC, self).__init__()
        self.road_map = observations.road_map
        self.ego = observations.ego
        self.others = observations.others

        self.safe_distance = 1  # m, safe distance when both vehicles are static
        self.max_acc = self.ego.max_acc  # m/s^2 acceleration limits
        self.max_speed = self.get_max_speed()  # m/s speed limits

    def get_max_speed(self):
        return min(self.ego.max_speed, self.ego.locate.max_speed)

    def collision_detection(self):
        pass


class Constant:

    def __init__(self):
        self.acc = 0

    def get_action(self):
        return self.acc
