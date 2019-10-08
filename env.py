import numpy as np
from enum import Enum
import random
import math
import uuid


class InterParam:
    """
    InterParam: parameters for constructing an intersection
    """
    x_min = -20.0
    x_max = 20.0
    y_min = -20.0
    inter_x = 0.0
    inter_y = 0.0
    inter_width = 6.0
    inter_height = 6.0
    line_width = inter_width / 3
    num_lanes = 2

    max_speed = 3.0  # m/s


class Segment:
    """
    Segment: a straight road
    """

    def __init__(self, start_x, start_y, length, angle, max_speed=3):
        """
        Constructor, define the start point, length, angle relative to coordinate and max speed
        :param start_x: start point in x axis
        :param start_y: start point in y axis
        :param length: length of road
        :param angle: angle relative to coordinate axis. The angle of x axis is 0
        :param max_speed: speed limits of the road
        """
        self.x = start_x
        self.y = start_y
        self.len = length
        self.angle = angle
        self.x_end = self.x + math.cos(self.angle) * self.len
        self.y_end = self.y + math.sin(self.angle) * self.len
        self.max_speed = max_speed

    def move(self, traveling_state, action, dt):
        """
        How car moving when running at straight segment. Part of transition model
        :param traveling_state: traveling state of this vehicle
        :param action: acceleration it will tack
        :param dt: time interval
        :return: new TravelingState
        """
        s = traveling_state.v + action * dt * dt / 2
        x = s * math.cos(self.angle) + traveling_state.x
        y = s * math.sin(self.angle) + traveling_state.y
        v = traveling_state.v + action * dt
        theta = self.angle
        return TravelingState(x, y, theta, v)

    def contains(self, x, y):
        """
        define whether a point is in the segment
        :param x: x
        :param y: y
        :return: bool, is in
        """
        epsilon = 1e-6
        if self.angle % math.pi == 0:
            # horizontal
            return (self.x - epsilon <= x <= self.x_end + epsilon) or (self.x_end - epsilon <= x <= self.x + epsilon)
        else:
            # vertical
            return (self.y - epsilon <= y <= self.y_end + epsilon) or (self.y_end - epsilon <= y <= self.y + epsilon)

    def get_random_point(self):
        """
        get a random point in this segment
        :return:
        """
        if self.angle % math.pi == 0:
            # horizontal
            return self.x + random.random() * (self.x_end - self.x), self.y
        else:
            return self.x, self.y + random.random() * (self.y_end - self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Connection:
    """
    Connection: connection of two segments. Its shape is 1/4 circle or a straight line.
    """

    def __init__(self, seg1, seg2, max_speed=2):
        """
        Constructor. Define the center, radius, left_turn of the connection
        :param seg1:
        :param seg2:
        """
        # save seg
        self.seg1 = seg1
        self.seg2 = seg2
        # radius of the connection, r==0 when it is a line
        self.r = max(seg1.y_end, seg2.y) - min(seg1.y_end, seg2.y)
        # left_turn
        if seg1.x_end < seg2.x:
            self.left_turn = -1
        else:
            self.left_turn = 1
        # center of circle
        if seg1.y_end < seg2.y:
            self.y = seg1.y_end
            self.x = seg2.x
        else:
            self.y = seg2.y
            self.x = seg1.x_end
        self.max_speed = max_speed

    def is_line(self):
        """
        judge whether a Connection connect two seg with a line
        :return: bool, is a line or not
        """
        return self.r == 0

    def contains(self, x, y):
        epsilon = 1e-6
        x_min = min(self.seg1.x_end, self.seg2.x) - epsilon
        x_max = max(self.seg1.x_end, self.seg2.x) + epsilon
        y_min = min(self.seg1.y_end, self.seg2.y) - epsilon
        y_max = max(self.seg1.y_end, self.seg2.y) + epsilon
        return x_min <= x <= x_max and y_min <= y <= y_max

    def move(self, traveling_state, action, dt):
        """
        car moving when running at intersection
        :param traveling_state: traveling state of this vehicle
        :param action: acceleration it will tack
        :param dt: time interval
        :return: new TravelingState
        """
        s = traveling_state.v * dt + action * dt * dt / 2
        # if connection is a line
        if self.is_line():
            x = traveling_state.x - self.left_turn * s
            v = traveling_state.v + action * dt
            y = traveling_state.y
            theta = traveling_state.theta
        else:
            d_eta = s / self.r
            eta = traveling_state.theta - self.left_turn * math.pi / 2 + self.left_turn * d_eta
            x = self.x + self.r * math.cos(eta)
            y = self.y + self.r * math.sin(eta)
            theta = traveling_state.theta + self.left_turn * d_eta
            v = traveling_state.v + action * dt
        return TravelingState(x, y, theta, v)


class RoadMap:
    """
    RoadMap: a static map that contains information of current environment
    """

    def __init__(self, segments, connections):
        self.segs = segments
        self.conns = connections

    def get_connection(self, seg1, seg2):
        for conn in self.conns:
            if seg1 == conn.seg1 and seg2 == conn.seg2:
                return conn
        return None

    def get_available_goals(self, seg):
        """
        get all available goals for a given segment
        :param seg: a start segment
        :return: available goal segments
        """
        available = []
        for conn in self.conns:
            if conn.seg1 == seg:
                available.append(conn.seg2)
        return available

    def get_available_starts(self):
        """
        get all available start segments
        :return: available start segments
        """
        available = []
        for conn in self.conns:
            available.append(conn.seg1)
        return available


def _gen_T_inter(params):
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
    # speed limit
    max_speed = params.max_speed

    seg1 = Segment(x_max, -line_width / 2, x_max - inter_x - inter_width / 2, math.pi, max_speed)
    seg2 = Segment(inter_x - inter_width / 2, -line_width / 2, inter_x - x_min - inter_width / 2, math.pi, max_speed)
    seg3 = Segment(x_min, -line_width / 2 * 3, inter_x - x_min - inter_width / 2, 0, max_speed)
    seg4 = Segment(inter_x + inter_width / 2, -line_width / 2 * 3, x_max - inter_x - inter_width / 2, 0, max_speed)
    seg5 = Segment(inter_x - inter_width / 2, -inter_height, inter_y - inter_height - y_min, -math.pi / 2, max_speed)
    seg6 = Segment(inter_x + inter_width / 2, y_min, inter_y - inter_height - y_min, math.pi / 2, max_speed)

    conn12 = Connection(seg1, seg2)
    conn34 = Connection(seg3, seg4)
    conn15 = Connection(seg1, seg5)
    conn35 = Connection(seg3, seg5)
    conn62 = Connection(seg6, seg2)
    conn64 = Connection(seg6, seg4)

    return RoadMap([seg1, seg2, seg3, seg4, seg5, seg6], [conn12, conn34, conn15, conn35, conn62, conn64])


class TravelingState:

    def __init__(self, x, y, theta, v):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v


class Vehicle:
    """
    a single vehicle's state and its attribute
    Each state has the following attribute:
        max_speed: speed limit of a road
        intent: intention, only the other vehicles have intention as a hidden attribute
    Attributes are: shape of the vehicle, action that can be take
    """

    def __init__(self, state, locate, goal, road_map, max_speed=16, max_acc=3):
        self.id = uuid.uuid4()
        self.state = state  # TravelingState
        self.max_speed = max_speed
        self.max_acc = max_acc  # m/s^2 max acceleration
        self.length = 2  # shape: length of a vehicle
        self.width = 1  # shape: width of a vehicle
        self.collide_range = math.sqrt(2) * self.length / 2  # collide detection range
        self.locate = locate  # which segment or connection
        self.goal = goal  # where to go
        self.map = road_map  # road map of current environment
        self.exist = True  # whether exists

    def collide(self, other):
        # using sphere model
        return self.get_distance(other) > 0

    def get_distance(self, other):
        return math.sqrt((self.state.x - other.state.x) ** 2 + (self.state.y - other.state.y) ** 2) \
               - self.collide_range - other.collide_range

    def set_shape(self, length, width):
        self.length = length
        self.width = width

    def step(self, action, dt):
        new_state = self.locate.move(self.state, action, dt)
        if not self.locate.contains(new_state.x, new_state.y):
            # move to a new segment, connection or out of the map
            if self.locate == self.goal:
                # out of map
                self.exist = False
                return
            elif self.goal.contains(new_state.x, new_state.y):
                # enter goal segment
                self.locate = self.goal
            else:
                # enter Connection
                conn = self.map.get_connection(self.locate, self.goal)
                self.locate = conn
        # update state
        self.state = new_state

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id


class Observations:
    """
    Observation: the scene that a driver can see
    """

    def __init__(self, env, vehicle):
        self.road_map = env.road_map  # static map
        self.ego = vehicle  # current vehicle
        self.others = []

        self.get_observation_single(env)  # get all other vehicles' observation

    def get_observation_single(self, env):
        for vehicle in env.state:
            if vehicle != self.ego:
                self.others.append(Observation(vehicle))


class Observation:

    def __init__(self, vehicle):
        self.travel_state = vehicle.state
        self.collision_range = vehicle.collide_range
        self.locate = vehicle.locate
        self.max_speed = vehicle.max_speed


class Env:

    def __init__(self, params):
        """
        init simulation environment.
        """
        self.params = params  # save params
        self.road_map = None  # map
        self.state = []  # state, that is a vehicle container
        self.observation = None
        self.action_space = []
        self.done = False  # done
        self.dt = 0.1  # s, interval
        self.reset()

    def reset(self):
        """
        reset the environment
        :return: observation after reset
        """
        self.road_map = _gen_T_inter(self.params)  # get map

        # === state ===
        self.state.clear()  # clean
        self.state.append(self._gen_vehicle_fix())  # add ego vehicle
        # add random other vehicles
        for i in range(2):
            self.state.append(self._gen_vehicle_random())
        self.action_space = [-1, 0, 1]

        # === observation ===
        self.observation = Observations(self, self.state[0])

        # === done ===
        self.done = False

        return self.observation

    def _gen_vehicle_random(self):
        """
        generate a vehicle in a random segment
        :return:
        """
        # get all segments that are not occupied
        available = []
        for seg in self.road_map.segs:
            contains = False
            for vehicle in self.state:
                if vehicle.locate == seg:
                    contains = True
                    break
            if not contains:
                available.append(seg)
        # generate a vehicle
        locate = random.choice(self.road_map.get_available_starts())
        x, y = locate.get_random_point()
        theta = locate.angle
        v = random.randrange(locate.max_speed)
        state = TravelingState(x, y, theta, v)
        goal = random.choice(self.road_map.get_available_goals(locate))
        vehicle = Vehicle(state, locate, goal, self.road_map)
        # add to state
        return vehicle

    def _gen_vehicle_fix(self):
        locate = self.road_map.segs[5]
        state = TravelingState(locate.x, locate.y, locate.angle, 1)
        goal = self.road_map.segs[1]
        return Vehicle(state, locate, goal, self.road_map)

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
        for vehicle in self.state:
            vehicle.step(action, self.dt)
            if not vehicle.exist:
                self.state.remove(vehicle)
        if not self.state:
            self.done = True
        return self.state, self.done

    def __del__(self):
        pass


class Intent(Enum):
    take_way = 1
    give_way = 2
    cautious = 3


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
        return min(self.K * (self.v_max - v))

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
