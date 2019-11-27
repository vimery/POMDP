from tools import *
import uuid


class State:
    """
    State: a class that contains the model of the environment
    """

    def __init__(self, ego=None, others=None):
        self.vehicles = [ego]
        if others:
            self.vehicles.extend(others)

    def reset(self):
        self.vehicles.clear()
        self.vehicles.append(None)

    def step(self, action, dt):
        # moving forward
        self.vehicles[0].step(action, dt)
        for other in self.vehicles[1:]:
            other.step(dt=dt)
            if not other.exist:
                self.vehicles.remove(other)

    def set_ego(self, ego):
        self.vehicles[0] = ego

    def add_others(self, other):
        self.vehicles.append(other)

    def get_vehicle_state(self):
        return [vehicle.state for vehicle in self.vehicles]


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


class Vehicle:

    def __init__(self, state, image, agent=None, max_speed=3, max_acc=1, length=2, width=1):
        """
        a single vehicle's state and its attribute
        :param state: VehicleState, contains x, y, theta, route, v
        :param image: image of this vehicle
        :param agent: planner of this vehicle
        :param length: length of the vehicle
        :param width: width of the vehicle
        :param max_speed: max design speed of the vehicle
        :param max_acc: max design acceleration of the vehicle
        """
        self.id = uuid.uuid4()
        self.state = state  # VehicleState
        self.agent = agent
        self.max_speed = max_speed
        self.max_acc = max_acc  # m/s^2 max acceleration
        self.length = length  # shape: length of a vehicle
        self.width = width  # shape: width of a vehicle
        self.collide_range = math.sqrt(2) * self.length / 2  # collide detection range
        self._image = loadImage(image)
        self.rect = None
        self.exist = True  # whether exists
        self._move_image()

    def collide(self, others):
        return self.rect.collidelist([other.rect for other in others]) != -1

    #
    # def get_distance(self, other):
    #     return math.sqrt((self.state.x - other.state.x) ** 2 + (self.state.y - other.state.y) ** 2) \
    #            - self.collide_range - other.collide_range

    def step(self, action=None, dt=1):
        if self.agent:
            action = self.agent.get_action(self.state)
        distance = self.state.v * dt + action * dt * dt / 2
        self.state.x, self.state.y, self.state.theta = self.state.route.next(self.state.x, self.state.y,
                                                                             self.state.theta, distance)
        self.state.v = self.state.v + action * dt
        if not self.state.x:
            self.exist = False

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def _move_image(self):
        image = pg.transform.rotate(self._image, self.state.theta / math.pi * 180)
        x, y = cartesian2py(self.state.x, self.state.y)
        self.rect = image.get_rect()
        self.rect = self.rect.move(x - self.rect.center[0], y - self.rect.center[1])
        return image

    def render(self, surface):
        if self.exist:
            surface.blit(self._move_image(), self.rect)


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
