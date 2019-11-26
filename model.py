class State:
    """
    State: a class that contains the model of the environment
    """

    def __init__(self, ego, others):
        self.ego = ego
        self.others = others


class VehicleState:
    """
    VehicleState: state of one vehicle
    """

    def __init__(self, x, y, theta, route, v):
        self.x = x
        self.y = y
        self.theta = theta
        self.route = route
        self.v = min(v, route.seg1.max_speed)

    def step(self, action):
        """
        move the vehicle with action along current route, will change current state
        :param action: Action, acceleration to be taken
        """
        pass


class Action:
    """
    Action: the action that a vehicle can take
    """

    def __init__(self):
        pass


class Observation:
    """
    Observation: observation of the current state, used to infer belief state
    """

    def __init__(self):
        pass
