import numpy as np
from enum import Enum
import random
import math


class InterParam:
    x_min = -20
    x_max = 20
    y_min = -20
    inter_x = 0.0
    inter_y = 0.0
    inter_width = 6.0
    inter_height = 6.0
    num_lanes = 2

    speed_limit = 10  # m/s


class Segment:

    def __init__(self, start_x, start_y, length, angle):
        self.x = start_x
        self.y = start_y
        self.len = length
        self.angle = angle

    def move(self, vehicle_state, action, dt):
        """
        car moving when running at straight segment
        :param vehicle_state: state of this vehicle
        :param action: acceleration it will tack
        :param dt: time interval
        :return: new vehicle_state
        """
        s = vehicle_state.v + action * dt * dt / 2
        x = s * math.cos(self.angle) + vehicle_state.x
        y = s * math.sin(self.angle) + vehicle_state.y
        v = vehicle_state.v + action * dt
        theta = self.angle
        return VehicleState(x, y, theta, v)

    def contains(self, vehicle_state):
        x = vehicle_state.x
        y = vehicle_state.y
        # horizontal
        x_end = self.x + math.cos(self.angle) * self.len
        y_end = self.y + math.sin(self.angle) * self.len
        if self.angle % math.pi == 0:
            return (self.x < x < x_end) or (x_end < x < self.x)
        else:
            return (self.y < y < y_end) or (y_end < y < self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Connection:

    def __init__(self, seg1, seg2):
        # save seg
        self.seg1 = seg1
        self.seg2 = seg2
        # radius of the connection, r==0 when it is a line
        self.r = max(seg1.y, seg2.y) - min(seg1.y, seg2.y)
        # order, clockwise: -1, anti-clockwise: 1
        if seg1.x < seg2.x:
            self.order = -1
        else:
            self.order = 1
        # center of circle
        if seg1.y < seg2.y:
            self.y = seg1.y
            self.x = seg2.x
        else:
            self.y = seg2.y
            self.x = seg1.x

    def is_line(self):
        """
        judge whether a Connection connect two seg with a line
        :return: bool, is a line or not
        """
        return self.r == 0

    def move(self, vehicle_state, action, dt):
        """
        car moving when running at intersection
        :param vehicle_state: state of this vehicle
        :param action: acceleration it will tack
        :param dt: time interval
        :return: new vehicle_state
        """
        s = vehicle_state.v * dt + action * dt * dt / 2
        # if connection is a line
        if self.is_line():
            x = vehicle_state.x - self.order * s
            v = vehicle_state.v + action * dt
            return VehicleState(x, vehicle_state.y, vehicle_state.theta, v)
        d_eta = s / self.r
        eta = vehicle_state.theta - self.order * math.pi / 2 + self.order * d_eta
        x = self.x + self.r * math.cos(eta)
        y = self.y + self.r * math.sin(eta)
        theta = vehicle_state.theta + self.order * d_eta
        v = vehicle_state.v + action * dt
        return VehicleState(x, y, theta, v)


class RoadMap:

    def __init__(self, segments, connections):
        self.segs = segments
        self.conns = connections

    def get_connection(self, seg):
        for conn in self.conns:
            if seg == conn.seg1:
                return seg
        return None


class Env:

    def __init__(self, params):
        """
        init simulation environment.
        """
        roadMap = self._gen_T_inter(params)

    def _gen_T_inter(self, params):
        """
        ——————————————————————————————————————————
            <---- seg2   inter     <---- seg1
        --------------           -----------------
            ----> seg3             ----> seg4
        ——————————————           —————————————————
                     |     |     |
                     | seg | seg |
                     |  5  |  6  |
                     |     |     |
                     |     |     |
                     |     |     |
        :param params:
        :return:
        """
        x_min = params.x_min
        x_max = params.x_max
        y_min = params.y_min
        # position of intersection, default is (0,0)
        inter_x = params.inter_x
        inter_y = params.inter_y
        # shape of intersection, default is square
        inter_width = params.inter_width
        inter_height = params.inter_height
        # shape of line, default is 1/2 * inter_width
        line_width = params.line_width

        seg1 = Segment(x_max, line_width / 2, x_max - inter_x - inter_width / 2, math.pi)
        seg2 = Segment(inter_x - inter_width / 2, line_width / 2, inter_x - x_min - inter_width / 2, math.pi)
        seg3 = Segment(x_min, line_width / 2 * 3, inter_x - x_min - inter_width / 2, 0)
        seg4 = Segment(inter_x + inter_width / 2, line_width / 2 * 3, x_max - inter_x - inter_width / 2, 0)
        seg5 = Segment(inter_x - inter_width / 2, -inter_height, inter_y - inter_height - y_min, -math.pi / 2)
        seg6 = Segment(inter_x + inter_width / 2, -inter_height, inter_y - inter_height - y_min, math.pi / 2)

        conn12 = Connection(seg1, seg2)
        conn34 = Connection(seg3, seg4)
        conn15 = Connection(seg1, seg5)
        conn35 = Connection(seg3, seg5)
        conn62 = Connection(seg6, seg2)
        conn64 = Connection(seg6, seg4)

        return RoadMap([seg1, seg2, seg3, seg4, seg5, seg6], [conn12, conn34, conn15, conn35, conn62, conn64])

    def reset(self):
        """
        reset the environment
        :return: observation after reset
        """
        self.state = State()
        self._compute_action()
        self.time = 0
        self.reward = 0.0
        self.done = False
        self.is_render = False
        return self.state

    def render(self):
        """
        show graphic image of simulator
        """
        # if not self.is_render:
        #     pygame.init()
        #     self.screen = pygame.display.set_mode((400, 400), 0, 32)
        # ego_loc, car_loc = self._transform()
        # ego = pygame.Rect(ego_loc)
        # car = pygame.Rect(car_loc)
        # self.is_render = True

    def _transition(self, action):
        # state change of ego vehicle
        self.state.ego.v += self.interval * action
        self.state.ego.pos += (self.interval ** 2) * action / 2
        # state change of other vehicle
        self.state.car.pos += self.state.car.v * self.interval
        # compute reward before update acceleration
        self.reward = self._get_reward__()
        self.state.ego.a = action

    def _transform(self):
        ego = self.state.ego
        car = self.state.car
        ego_loc = ((-ego.width / 2, ego.pos), (ego.width / 2, ego.pos - ego.length))
        car_loc = ((car.pos - car.length, car.width / 2), (car.pos, -car.width / 2))
        return ego_loc, car_loc

    def _compute_action(self):
        action = Action()
        state = self.state
        self.actions.clear()
        # self.actions.append(-1)
        self.actions.append(0)
        self.actions.append(1)
        # self.actions.append(action.keep_speed(state.ego.v))
        # self.actions.append(action.keep_distance(state.ego, state.car))
        # self.actions[2] = action.keep_distance(state.ego, state.cars[0])
        # self.actions[3] = action.keep_distance(state.ego, state.cars[1])
        # self.actions[4] = action.keep_distance(state.ego, state.cars[2])
        # self.actions[5] = action.keep_distance(state.ego, state.cars[3])

    def _get_reward__(self):
        self.done = True
        if self._success():
            return 1.0 - self.time / self.max_time
        if self._failure():
            return -2.0
        if self._time_out():
            return -0.1
        self.done = False
        return 0

    def _success(self):
        ego = self.state.ego
        # success when the ego vehicle pass the intersection
        if ego.pos - ego.length > ego.delta:
            return True
        return False

    def _failure(self):
        # collide
        if self.state.ego.collide(self.state.car):
            return True
        return False

    def _time_out(self):
        if self.time > self.max_time:
            return True
        return False

    # def _done(self):
    #     return self._success() or not self._failure()

    def step(self, action):
        """
        taken an action and return the simulation results
        :param action: action needs to be taken
        :return:
        Observation: new observation after taking an action
        reward,
        done: the simulation is finished,
        info: debug message
        """
        self._transition(action)
        return self.state, self.reward, self.done

    def __del__(self):
        pass


class Intent(Enum):
    take_way = 1
    give_way = 2
    cautious = 3


class Vehicle:
    """
    a single vehicle's state and its attribute
    Each state has the following attribute:
        x, y, theta: pose of the vehicle
        v: speed
        a: acceleration
        max_speed: speed limit of a road
        delta: distance from stop line to the center of the intersection
        intent: intention, only the other vehicles have intention as a hidden attribute
    Attributes are: shape of the vehicle, action that can be take
    """

    def __init__(self, x=0, y=0, theta=0, v=0, a=0, max_speed=30, delta=4, intent=Intent.cautious):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.a = a
        self.delta = delta
        self.intent = intent
        self.max_speed = max_speed
        self.length = 2
        self.width = 1
        self.action = None

    def collide(self, other):
        # using sphere model
        dis = np.sqrt(np.sum(np.square(self.x - other.x), np.square(self.y - other.y)))
        if dis > self.length + other.length:
            return False
        return True

    def step(self, action, dt):
        pass


class VehicleState:

    def __init__(self, x=0, y=0, theta=0, v=0, locate=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.locate = locate


class State(object):
    """
    State represents the whole environment including four vehicles' state and the ego vehicle's state.
    """

    def __init__(self):
        self.ego = Vehicle(-random.randint(4, 10))
        self.car = Vehicle(-random.randint(4, 10), 10)
        # self.cars = []
        # self.cars_count = 0  # random.randint(1, 4)
        # for i in range(self.cars_count):
        #     self.cars[i] = Vehicle(random.randint(4, 10), 0, 0, 4, random.randint(1, 3))
        #     self.cars[i].action =


class Observation:
    """
    Observation is the visible part that the ego vehicle can observe from current state.
    In detail, observation contains the position, speed, acceleration and start of intersection of every vehicle
    and the predicted acceleration of ego vehicle in each action
    """

    def __init__(self, pos=10, v=0, a=0, delta=4):
        self.pos = pos
        self.v = v
        self.a = a
        self.delta = delta


class Observations:

    def __init__(self, state):
        self.state = state


class Action:

    def __init__(self, K=1, v_max=50, c1=1, c2=1, mu=0.5):
        self.K = K
        self.v_max = v_max
        self.c1 = c1
        self.c2 = c2
        self.mu = mu

    def keep_speed(self, v):
        """
        compute acceleration to keep set speed
        :param v: speed of ego vehicle
        :return: acceleration to keep set speed
        """
        return min(self.K * (self.v_max - v), max_acc)

    def keep_distance(self, ego, target):
        """
        compute acceleration to keep distance to target vehicle
        :param ego: ego vehicle
        :param target: target vehicle
        :return: acceleration to keep distance to target vehicle
        """
        x1 = target.pos - ego.pos
        x2 = target.v - ego.v
        return (-self.c1 * x2 + self.mu * np.sign(self.c1 * x1 + self.c2 * x2)) / self.c2
