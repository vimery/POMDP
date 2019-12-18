import uuid

from Sim.tools import *


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
        self.radius = math.sqrt(length ** 2 + width ** 2)  # actual height and width is 2 times in image
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
        self.x, self.y, self.theta, self.v, self.action = self.forward(action, dt)
        if not self.x:
            self.exist = False

    def forward(self, action, dt):
        # check acc limit
        if action > self.max_acc:
            action = self.max_acc
        elif action < self.min_acc:
            action = self.min_acc
        # check speed limit
        v = self.v
        max_v = self.get_max_speed()
        if v + action * dt > max_v:
            acc_t = (max_v - v) / action
            acc_distance = (max_v + v) * acc_t / 2
            distance = acc_distance + max_v * (dt - acc_t)
            v = max_v
        elif v + action * dt < 0:
            mod_t = v / action
            distance = v * mod_t / 2
            v = 0
        else:
            distance = self.v * dt + action * dt * dt / 2
            v = self.v + action * dt
        x, y, theta = self.route.next(self.x, self.y, self.theta, distance)
        return x, y, theta, v, action

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
            if i in self.vehicles and self.vehicles[i].exist:
                if i != self.id:
                    vehicle = self.vehicles[i]
                    ob.append([vehicle.x, vehicle.y, vehicle.theta, vehicle.v])
            else:
                ob.append([0, 0, 0, 0])
        return ob
