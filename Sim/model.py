import uuid

from Sim.tools import *


class State:
    """
    State: a class that contains the model of the environment
    """

    def __init__(self, ego=None, others=None, agents=None):
        self.vehicles = [ego]
        self.agents = {}
        if others:
            if not agents or len(agents) != others:
                raise (Exception("must give an agent to a non-autonomous vehicle"))
            for i in range(len(others)):
                self.agents[others[i].id] = agents[i]
            self.vehicles.extend(others)

    def reset(self):
        if self.agents:
            self.agents.clear()
        self.vehicles.clear()
        self.vehicles.append(None)

    def step(self, actions, dt):
        if len(actions) != len(self.vehicles):
            raise Exception("not all vehicles have an action")
        # moving forward
        for i in range(0, len(self.vehicles)):
            self.vehicles[i].step(actions[i], dt=dt)
        for other in self.vehicles[1:]:
            if not other.exist:
                self.vehicles.remove(other)
                del self.agents[other.id]

    def set_ego(self, ego):
        self.vehicles[0] = ego

    def add_others(self, other, agent):
        self.vehicles.append(other)
        self.agents[other.id] = agent

    def get_vehicle_by_id(self, v_id):
        for vehicle in self.vehicles[1:]:
            if vehicle.id == v_id:
                return vehicle


class VehicleState:
    """
    VehicleState: state of one vehicle
    """

    def __init__(self, x, y, theta, v):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Vehicle:

    def __init__(self, route, v, image_name, max_speed=10, max_acc=1, min_acc=-5, length=2, width=1):
        self.id = uuid.uuid4()
        self.route = route
        self.state = VehicleState(route.seg1.x, route.seg1.y, route.seg1.theta, min(v, route.seg1.max_speed, max_speed))
        self.max_acc = max_acc
        self.min_acc = min_acc
        self.max_speed = max_speed
        self.length = length
        self.width = width
        self.radius = math.sqrt((length / 2) ** 2 + (width / 2) ** 2)  # for collide detection
        self.action = 0
        # image and rect setting
        self._image = load_image(image_name, width=width, height=length)
        self.rect = None
        self.exist = True  # whether exists

    def collide(self, others):
        for other in others:
            if collide_detection(self.state.x, self.state.y, other.state.x, other.state.y, self.radius, other.radius):
                return True
        return False

    def step(self, action, dt=1):
        """
        actual moving
        :param action: action to be taken
        :param dt: time that the action last
        """
        # check acc limit
        if action > self.max_acc:
            action = self.max_acc
        elif action < self.min_acc:
            action = self.min_acc
        self.action = action
        # check speed limit
        v = self.state.v
        max_v = self.get_max_speed()
        if v + action * dt > max_v:
            acc_t = (max_v - v) / action
            acc_distance = (max_v + v) * acc_t / 2
            distance = acc_distance + max_v * (dt - acc_t)
            self.state = self.forward(distance=distance)
            if self.state:
                self.state.v = max_v
        elif v + action * dt < 0:
            mod_t = v / action
            distance = v * mod_t / 2
            self.state = self.forward(distance=distance)
            if self.state:
                self.state.v = 0
        else:
            self.state = self.forward(action, dt)
        if not self.state:
            self.exist = False

    def forward(self, action=None, dt=None, distance=None):
        """
        simulate to move
        :param action:
        :param dt:
        :param distance:
        :return: new state or None if out of map
        """
        if distance is None:
            if action is None or dt is None:
                raise Exception("must set distance or (action and dt)")
            distance = self.state.v * dt + action * dt * dt / 2
        x, y, theta = self.route.next(self.state.x, self.state.y, self.state.theta, distance)
        if x is None:
            return None
        v = self.state.v + action * dt if action else self.state.v

        return VehicleState(x, y, theta, v)

    def get_max_speed(self):
        return min(self.max_speed, self.route.get_max_speed(self.state.x, self.state.y))

    def get_observation(self):
        return SingleOb(self.state, self.route, self.radius, self.action, self.max_acc, self.get_max_speed())

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def render(self, surface):
        if self.exist:
            image = pg.transform.rotate(self._image, self.state.theta / math.pi * 180)
            x, y = cartesian2py(self.state.x, self.state.y)
            rect = image.get_rect()
            rect = rect.move(x - rect.center[0], y - rect.center[1])
            surface.blit(image, rect)


class Action:
    """
    Action: the action that a vehicle can take
    """

    def __init__(self):
        pass


class SingleOb:

    def __init__(self, state, route, radius, a, max_acc, max_speed):
        self.state = state
        self.route = route
        self.radius = radius
        self.a = a
        self.max_acc = max_acc
        self.max_speed = max_speed


class Observation:
    """
    Observation: observation of the current state, used to infer belief state
    """

    def __init__(self, ego, others, road_map):
        self.ego = ego
        self.others = others
        self.road_map = road_map

    def get_array(self):
        e = self.ego.state
        ob = [e.x, e.y, e.theta, e.v]
        for v in self.others:
            s = v.state
            ob.extend([s.x, s.y, s.theta, s.v])
        for i in range(12 - len(ob)):
            ob.append(0)
        return ob
