import math
import random
from vehicle import TravelingState
from tools import *
import pygame as pg


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
        s = traveling_state.v * dt + action * dt * dt / 2
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

    def _get_line(self, top=-1):
        h = self.horizon
        v = not h
        w = self.width / 2
        x, y = self.x + top * v * w, self.y + top * h * w
        x_end, y_end = self.x_end + top * v * w, self.y_end + top * h * w
        return cartesian2py(Point(x, y)), cartesian2py(Point(x_end, y_end))

    def render(self, surface):
        start_1, end_1 = self._get_line(-1)
        start_2, end_2 = self._get_line(1)
        pg.draw.line(surface, Color.black, (start_1.x, start_1.y), (end_1.x, end_1.y), 2)
        pg.draw.line(surface, Color.black, (start_2.x, start_2.y), (end_2.x, end_2.y), 2)


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
        return self.r < epsilon

    def contains(self, x, y):
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
        if traveling_state.v == 0 and action == 0:
            return traveling_state
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
