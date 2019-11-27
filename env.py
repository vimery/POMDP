#!/usr/bin/env python3

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
from agent import *
from model import *
from roadmap import *
import random

# max step number
max_steps = 1000


def _gen_t_inter(params):
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
    y_max = params.y_max
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


def _gen_full_inter(params):
    """
                     |     |     |
                     |  7  |  8  |
                     | seg | seg |
                     |     |     |
                     |     |     |
        ——————————————           —————————————————
            <---- seg2             <---- seg1
        --------------   inter   -----------------
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
    line_width = params.inter_width / 2
    # horizon
    r_start_x = params.x_max
    r_length = params.x_max - line_width
    l_start_x = params.x_min
    l_length = -params.x_min - line_width
    # vertical
    d_start_y = params.y_min
    d_length = -params.y_min - line_width
    t_start_y = params.y_max
    t_length = params.y_max - line_width

    p = line_width / 2
    seg1 = Segment(r_start_x, p, r_length, line_width, pi)
    seg2 = Segment(-line_width, p, l_length, line_width, pi)
    seg3 = Segment(l_start_x, -p, l_length, line_width, 0)
    seg4 = Segment(line_width, -p, r_length, line_width, 0)
    seg5 = Segment(-p, -line_width, d_length, line_width, nh_pi)
    seg6 = Segment(p, d_start_y, d_length, line_width, h_pi)
    seg7 = Segment(-p, t_start_y, t_length, line_width, nh_pi)
    seg8 = Segment(p, line_width, t_length, line_width, h_pi)

    routes = [Route(seg1, Connection(seg1, seg2), seg2), Route(seg1, Connection(seg1, seg5), seg5),
              Route(seg1, Connection(seg1, seg8), seg8), Route(seg3, Connection(seg3, seg5), seg5),
              Route(seg3, Connection(seg3, seg4), seg4), Route(seg3, Connection(seg3, seg8), seg8),
              Route(seg6, Connection(seg6, seg4), seg4), Route(seg6, Connection(seg6, seg2), seg2),
              Route(seg6, Connection(seg6, seg8), seg8), Route(seg7, Connection(seg7, seg2), seg2),
              Route(seg7, Connection(seg7, seg4), seg4), Route(seg7, Connection(seg7, seg5), seg5)]

    return RoadMap(routes)


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
        self.road_map = _gen_full_inter(params)  # get map
        self.state = State()
        self.observation = None
        self.action_space = []
        self.dt = 0.1  # s, interval
        self.step_count = 0
        # pygame
        pg.init()
        self.scale = params.scale
        self.size_x = (params.x_max - params.x_min) * self.scale
        self.size_y = (params.y_max - params.y_min) * self.scale
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
        self.state.reset()  # clean
        self.state.set_ego(self._gen_vehicle_fix())
        # add random other vehicles
        for i in range(2):
            other = self._get_random_vehicle()
            if other:
                self.state.add_others(other)
        self.action_space = [-1, 0, 1]

        # === pygame ===
        self.render()

        return self.get_observation()

    def _get_random_vehicle(self):
        """
        generate a vehicle in a random route
        :return:
        """
        # generate a vehicle
        vehicle = None
        count = 0
        # try up to 5 times
        while not vehicle and count < 5:
            route = random.choice(self.road_map.routes)
            x, y, theta = route.seg1.x, route.seg1.y, route.seg1.theta
            v = random.randrange(1, route.seg1.max_speed)
            state = VehicleState(x, y, theta, route, v)
            agent = Constant()
            vehicle = Vehicle(state, "other_vehicle.png", agent)
            if vehicle.collide(self.state.vehicles):
                vehicle = None
            count = count + 1
        # add to state
        return vehicle

    def _gen_vehicle_fix(self):
        route = self.road_map.routes[11]
        state = VehicleState(route.seg1.x, route.seg1.y, route.seg1.theta, route, 3)
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
        for vehicle in self.state.vehicles:
            vehicle.render(self.win)
        # quit event
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                exit(0)
        pg.display.update()
        pg.time.wait(100)

    def _draw_background(self, back_color):
        self.win.fill(back_color, self.background)

    def get_observation(self):
        return self.state.get_vehicle_state()

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
        # add random vehicle with rate 0.1
        if random.random() < 0.01:
            other = self._get_random_vehicle()
            if other:
                self.state.add_others(other)
        self.state.step(action, self.dt)
        self.step_count += 1
        ego = self.state.vehicles[0]
        others = self.state.vehicles[1:]
        if ego.exist:
            if others and ego.collide(others):
                done = -1  # collide
            else:
                done = 0  # normal
        else:
            done = 1  # out of map
        if not done and self.step_count > max_steps:
            done = -2  # overtime
        return self.get_observation(), done, self.step_count

    def __del__(self):
        pass
