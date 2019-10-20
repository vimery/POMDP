import pygame as pg
import uuid
from tools import loadImage
from tools import cartesian2py
from tools import Point
import math


class TravelingState:

    def __init__(self, x, y, theta, v):
        """
        A vehicle's moving state, including position, angle and speed
        :param x: center point x of this vehicle
        :param y: center point y of this vehicle
        :param theta: angle
        :param v: speed
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v


class Vehicle:

    def __init__(self, state, locate, goal, road_map, image, max_speed=16, max_acc=3):
        """
        a single vehicle's state and its attribute
        :param state: Travel state, contains x, y, theta, v
        :param locate: a segment or a connection
        :param goal: goal segment
        :param road_map: road map
        :param image: image of this vehicle
        :param max_speed: max design speed of the vehicle
        :param max_acc: max design acceleration of the vehicle
        """
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
        self.rect = self.image.get_rect()
        self.exist = True  # whether exists

    def collide(self, others):
        return self.rect.collidelist([other.rect for other in others]) != -1
    #
    # def get_distance(self, other):
    #     return math.sqrt((self.state.x - other.state.x) ** 2 + (self.state.y - other.state.y) ** 2) \
    #            - self.collide_range - other.collide_range

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
        top = cartesian2py(Point(self.state.x, self.state.y))
        self.rect = image.get_rect()
        self.rect = self.rect.move(top.x - self.rect.center[0], top.y - self.rect.center[1])
        surface.blit(image, self.rect)
