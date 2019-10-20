#!/usr/bin/env python3

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import numpy as np
from enum import Enum
import random
import math
from tools import *
from vehicle import Vehicle
from vehicle import TravelingState
from roadmap import RoadMap
from roadmap import Segment
from roadmap import Connection


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
        self.dt = 0.5  # s, interval
        self.count = 0  # step count1
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
        self.state.append(self._gen_vehicle_fix())  # add ego vehicle, at 0
        # add random other vehicles
        for i in range(2):
            self.state.append(self._gen_vehicle_random())
        self.action_space = [-1, 0, 1]

        # === observation ===
        self.observation = Observations(self, self.state[0])

        # === pygame ===
        self.render()

        return self.observation

    def _gen_vehicle_random(self):
        """
        generate a vehicle in a random segment
        :return:
        """
        # get all segments that are not occupied
        available = self.road_map.get_available_starts()
        for seg in available:
            for vehicle in self.state:
                if vehicle.locate == seg:
                    available.remove(seg)
        # generate a vehicle
        locate = random.choice(available)
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
        pg.time.wait(100)

    def _draw_background(self, back_color):
        self.win.fill(back_color, self.background)

    def success(self):
        if not self.state:
            return True
        return False

    def over_time(self):
        return self.count > max_steps

    def collide(self):
        return self.state[0].collide(self.state[1:])

    def done(self):
        if self.success():
            return 1
        elif self.collide():
            return -1
        elif self.over_time():
            return -2
        return False
        # return self.success() or self.over_time() or self.collide()

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
        self.count += 1
        return self.state, self.done()

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
