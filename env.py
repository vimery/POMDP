#!/usr/bin/env python3

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import numpy as np
from enum import Enum
import random
import math
import uuid
from tools import *

try:
    import pygame as pg
except ImportError:
    render = False


class Segment:
    """
    Segment: a straight road
    """

    def __init__(self, start_x, start_y, length, width, angle, max_speed=3):
        """
        Constructor, define the start point, length, angle relative to coordinate and max speed
        :param start_x: start point in x axis
        :param start_y: start point in y axis
        :param length: length of road
        :param width: width of road
        :param angle: angle relative to coordinate axis. The angle of x axis is 0
        :param max_speed: speed limits of the road
        """
        self.x = start_x
        self.y = start_y
        self.width = width
        self.len = length
        self.angle = angle
        self.x_end = self.x + math.cos(self.angle) * self.len
        self.y_end = self.y + math.sin(self.angle) * self.len
        self.max_speed = max_speed
        self.horizon = self.angle % math.pi == 0

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
        if self.horizon:
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

    def render(self, surface):
        if self.horizon:
            start_1 = cartesian2py(Point(self.x, self.y - self.width / 2))
            start_2 = cartesian2py(Point(self.x, self.y + self.width / 2))
            end_1 = cartesian2py(Point(self.x_end, self.y_end - self.width / 2))
            end_2 = cartesian2py(Point(self.x_end, self.y_end + self.width / 2))
        else:
            start_1 = cartesian2py(Point(self.x - self.width / 2, self.y))
            start_2 = cartesian2py(Point(self.x + self.width / 2, self.y))
            end_1 = cartesian2py(Point(self.x_end - self.width / 2, self.y_end))
            end_2 = cartesian2py(Point(self.x_end + self.width / 2, self.y_end))
        pg.draw.line(surface, Color.black, start_1, end_1, 2)
        pg.draw.line(surface, Color.black, start_2, end_2, 2)


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

    def render(self, surface):
        pass


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

    def render(self, surface):
        for seg in self.segs:
            seg.render(surface)
        for conn in self.conns:
            conn.render(surface)


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

    seg1 = Segment(x_max, -line_width / 2, x_max - inter_x - inter_width / 2, line_width, math.pi, max_speed)
    seg2 = Segment(inter_x - inter_width / 2, -line_width / 2, inter_x - x_min - inter_width / 2, line_width, math.pi,
                   max_speed)
    seg3 = Segment(x_min, -line_width / 2 * 3, inter_x - x_min - inter_width / 2, line_width, 0, max_speed)
    seg4 = Segment(inter_x + inter_width / 2, -line_width / 2 * 3, x_max - inter_x - inter_width / 2, line_width, 0,
                   max_speed)
    seg5 = Segment(inter_x - inter_width / 2 + line_width / 2, -inter_height, inter_y - inter_height - y_min,
                   line_width, -math.pi / 2,
                   max_speed)
    seg6 = Segment(inter_x + inter_width / 2 - line_width / 2, y_min, inter_y - inter_height - y_min, line_width,
                   math.pi / 2, max_speed)

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

    def __init__(self, state, locate, goal, road_map, image, max_speed=16, max_acc=3):
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
        self.image = loadImage(image)
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

    def render(self, surface):
        image = pg.transform.rotate(self.image, self.state.theta / math.pi * 180)


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
        # pygame
        pg.init()
        self.scale = 10
        self.size_x = (params.x_max - params.x_min) * self.scale
        self.size_y = - params.y_min * self.scale
        self.win = pg.display.set_mode((self.size_x, self.size_y))
        pg.display.set_caption("Autonomous driving simulator")
        self.background = self.win.fill(Color.white, (0, 0, self.size_x, self.size_y))
        self.font = pg.font.SysFont("arial", 16)
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

        # === pygame ===
        self.render()

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
        v = random.randrange(1, locate.max_speed)
        state = TravelingState(x, y, theta, v)
        goal = random.choice(self.road_map.get_available_goals(locate))
        vehicle = Vehicle(state, locate, goal, self.road_map, "other_vehicle.png")
        # add to state
        return vehicle

    def _gen_vehicle_fix(self):
        locate = self.road_map.segs[5]
        state = TravelingState(locate.x, locate.y, locate.angle, 1)
        goal = self.road_map.segs[1]
        return Vehicle(state, locate, goal, self.road_map, "vehicle.png")

    def render(self):
        """
        show graphic image of simulator
        """
        # erase window
        self._draw_background(Color.white)
        # draw road_map
        self.road_map.render(self.win)
        # draw vehicles
        for vehicle in self.state:
            vehicle.render(self.win)
        pg.display.update()

    def _draw_background(self, color):
        self.win.fill(color, self.background)

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
