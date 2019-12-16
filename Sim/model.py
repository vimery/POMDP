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


class Vehicle:

    def __init__(self, route, v, v_id=uuid.uuid4(), agent=None, image_name=None, max_speed=10, max_acc=1, min_acc=-5,
                 length=2, width=1):
        self.id = v_id
        # dynamic
        self.x = route.seg1.x
        self.y = route.seg1.y
        self.theta = route.seg1.theta
        self.v = min(v, route.seg1.max_speed, max_speed)
        self.route = route
        self.action = 0

        # vehicle property
        self.max_acc = max_acc
        self.min_acc = min_acc
        self.max_speed = max_speed
        self.length = length
        self.width = width
        self.radius = math.sqrt((length / 2) ** 2 + (width / 2) ** 2)  # for collide detection
        self.agent = agent
        self.exist = True

        # image and rect setting
        self._need_render = False
        self._image_name = image_name
        self._image = None

    def collide(self, others):
        for other in others:
            if collide_detection(self.x, self.y, other.x, other.y, self.radius, other.radius):
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
        v = self.v
        max_v = self.get_max_speed()
        if v + action * dt > max_v:
            acc_t = (max_v - v) / action
            acc_distance = (max_v + v) * acc_t / 2
            distance = acc_distance + max_v * (dt - acc_t)
        elif v + action * dt < 0:
            mod_t = v / action
            distance = v * mod_t / 2
        else:
            distance = self.v * dt + action * dt * dt / 2
        self.x, self.y, self.theta = self.route.next(self.x, self.y, self.theta, distance)
        self.v = self.v + action * dt
        if not self.x:
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
            distance = self.v * dt + action * dt * dt / 2
        x, y, theta = self.route.next(self.x, self.y, self.theta, distance)
        v = self.v + action * dt if action else self.v

        return x, y, theta, v

    def get_max_speed(self):
        return min(self.max_speed, self.route.get_max_speed(self.x, self.y))

    def render(self, surface):
        if not self._need_render:
            self._need_render = True
            self._image = load_image(self._image_name, width=self.width, height=self.length)
        if self.exist:
            image = pg.transform.rotate(self._image, self.theta / math.pi * 180)
            x, y = cartesian2py(self.x, self.y)
            rect = image.get_rect()
            rect = rect.move(x - rect.center[0], y - rect.center[1])
            surface.blit(image, rect)


class Observation:
    """
    Observation: observation of the current state, used to infer belief state
    """

    def __init__(self, v_id, vehicles, count):
        self.id = v_id
        self.ego = vehicles[v_id]
        self.vehicles = vehicles
        self.count = count

    def get_array(self):
        ob = [[self.ego.x, self.ego.y, self.ego.theta, self.ego.v]]
        for i in range(self.count):
            if i in self.vehicles:
                if i != self.id:
                    vehicle = self.vehicles[i]
                    ob.append([vehicle.x, vehicle.y, vehicle.theta, vehicle.v])
            else:
                ob.append([0, 0, 0, 0])
        return ob
