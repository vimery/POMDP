#!/usr/bin/env python3

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import random
from enum import Enum

from roadmap import Connection
from roadmap import RoadMap
from roadmap import Route
from roadmap import Segment
from tools import *
from model import *
from vehicle import Vehicle

# max step number
max_steps = 1000


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

    route12 = Route(seg1, conn12, seg2)
    route34 = Route(seg3, conn34, seg4)
    route15 = Route(seg1, conn15, seg5)
    route35 = Route(seg3, conn35, seg5)
    route62 = Route(seg6, conn62, seg2)
    route64 = Route(seg6, conn64, seg4)

    return RoadMap([route12, route34, route15, route35, route62, route64])


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
        self.road_map = _gen_T_inter(params)  # get map
        self.vehicles = []
        self.observation = None
        self.action_space = []
        self.dt = 0.2  # s, interval
        self.step_count = 0
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
        # === state ===
        self.vehicles.clear()  # clean
        self.vehicles.append(self._gen_vehicle_fix())  # add ego vehicle, at 0
        # add random other vehicles
        # for i in range(2):
        #     self.vehicles.append(self._get_random_vehicle())
        self.action_space = [-1, 0, 1]

        # === pygame ===
        self.render()

        return self.observation

    def _get_random_vehicle(self):
        """
        generate a vehicle in a random route
        :return:
        """
        # get all routes that are not occupied
        available = self.road_map.routes
        for route in available:
            if route == route:
                available.remove(route)
        # generate a vehicle
        locate = random.choice(available)
        x, y = locate.get_random_point()
        theta = locate.angle
        v = random.randrange(1, locate.max_speed)
        state = VehicleState(x, y, theta, v)
        vehicle = Vehicle(state, "other_vehicle.png")
        # add to state
        return vehicle

    def _gen_vehicle_fix(self):
        route = self.road_map.routes[4]
        state = VehicleState(route.seg1.x, route.seg1.y, route.seg1.theta, route, 1)
        return Vehicle(state, "vehicle.png")

    def render(self):
        """
        show graphic image of simulator
        """
        # erase window
        self._draw_background(Color.white)
        # draw road_map
        self.road_map.render(self.win)
        # draw vehicles
        for vehicle in self.vehicles:
            vehicle.render(self.win)
        # quit event
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
        pg.display.update()
        pg.time.wait(100)

    def _draw_background(self, back_color):
        self.win.fill(back_color, self.background)

    def success(self):
        if not self.vehicles:
            return True
        return False

    def over_time(self):
        return self.step_count > max_steps

    def collide(self):
        return self.vehicles[0].collide(self.vehicles[1:])

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
        for vehicle in self.vehicles:
            vehicle.step(action, self.dt)
            if not vehicle.exist:
                self.vehicles.remove(vehicle)
        self.step_count += 1
        return self.vehicles, self.done()

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
