import pygame as pg
import uuid
from tools import loadImage
from tools import cartesian2py
from model import VehicleState
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

    def __init__(self, state, image, max_speed=3, max_acc=1, length=2, width=1):
        """
        a single vehicle's state and its attribute
        :param state: VehicleState, contains x, y, theta, route, v
        :param image: image of this vehicle
        :param length: length of the vehicle
        :param width: width of the vehicle
        :param max_speed: max design speed of the vehicle
        :param max_acc: max design acceleration of the vehicle
        """
        self.id = uuid.uuid4()
        self.state = state  # VehicleState
        self.max_speed = max_speed
        self.max_acc = max_acc  # m/s^2 max acceleration
        self.length = length  # shape: length of a vehicle
        self.width = width  # shape: width of a vehicle
        self.collide_range = math.sqrt(2) * self.length / 2  # collide detection range
        self.image = loadImage(image)
        self.rect = self.image.get_rect()
        self.exist = True  # whether exists

    def collide(self, others):
        return self.rect.collidelist([other.rect for other in others]) != -1

    #
    # def get_distance(self, other):
    #     return math.sqrt((self.state.x - other.state.x) ** 2 + (self.state.y - other.state.y) ** 2) \
    #            - self.collide_range - other.collide_range

    def step(self, action, dt):
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

    def render(self, surface):
        image = pg.transform.rotate(self.image, self.state.theta / math.pi * 180)
        x, y = cartesian2py(self.state.x, self.state.y)
        self.rect = image.get_rect()
        self.rect = self.rect.move(x - self.rect.center[0], y - self.rect.center[1])
        surface.blit(image, self.rect)
