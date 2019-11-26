class State:
    """
    State: a class that contains the model of the environment
    """
    def __init__(self):
        pass


class VehicleState:
    """
    VehicleState: state of one vehicle
    """
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.route = None
        self.v = 0.0

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
