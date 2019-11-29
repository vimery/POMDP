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


def _gen_segments(params):
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

    s = params.max_speed

    p = line_width / 2
    seg1 = Segment("1", r_start_x, p, r_length, line_width, pi, s)
    seg2 = Segment("2", -line_width, p, l_length, line_width, pi, s)
    seg3 = Segment("3", l_start_x, -p, l_length, line_width, 0, s)
    seg4 = Segment("4", line_width, -p, r_length, line_width, 0, s)
    seg5 = Segment("5", -p, -line_width, d_length, line_width, nh_pi, s)
    seg6 = Segment("6", p, d_start_y, d_length, line_width, h_pi, s)
    seg7 = Segment("7", -p, t_start_y, t_length, line_width, nh_pi, s)
    seg8 = Segment("8", p, line_width, t_length, line_width, h_pi, s)

    return seg1, seg2, seg3, seg4, seg5, seg6, seg7, seg8


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
    s = params.max_speed
    seg1, seg2, seg3, seg4, seg5, seg6, seg7, seg8 = _gen_segments(params)

    routes = [Route(seg1, Connection(seg1, seg2, s), seg2), Route(seg1, Connection(seg1, seg5, s), seg5),
              Route(seg3, Connection(seg3, seg5, s), seg5), Route(seg3, Connection(seg3, seg4, s), seg4),
              Route(seg6, Connection(seg6, seg4, s), seg4), Route(seg6, Connection(seg6, seg2, s), seg2)]

    return RoadMap(routes)


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
    s = params.max_speed
    seg1, seg2, seg3, seg4, seg5, seg6, seg7, seg8 = _gen_segments(params)

    routes = [Route(seg1, Connection(seg1, seg2, s), seg2), Route(seg1, Connection(seg1, seg5, s), seg5),
              Route(seg1, Connection(seg1, seg8, s), seg8), Route(seg3, Connection(seg3, seg5, s), seg5),
              Route(seg3, Connection(seg3, seg4, s), seg4), Route(seg3, Connection(seg3, seg8, s), seg8),
              Route(seg6, Connection(seg6, seg4, s), seg4), Route(seg6, Connection(seg6, seg2, s), seg2),
              Route(seg6, Connection(seg6, seg8, s), seg8), Route(seg7, Connection(seg7, seg2, s), seg2),
              Route(seg7, Connection(seg7, seg4, s), seg4), Route(seg7, Connection(seg7, seg5, s), seg5)]

    return RoadMap(routes)


class Env:

    def __init__(self, params):
        self.road_map = _gen_full_inter(params)  # get map
        self.state = State()
        self.action_space = []
        self.dt = 0.1  # s, interval
        self.step_count = 0

        # pygame
        pg.init()
        size_x = (params.x_max - params.x_min) * params.scale
        size_y = (params.y_max - params.y_min) * params.scale
        self.win = pg.display.set_mode((size_x, size_y))
        pg.display.set_caption("Autonomous driving simulator")
        self.background = self.win.fill(Color.white, (0, 0, size_x, size_y))
        self.font = pg.font.SysFont("arial", 16)

        self.reset()

    def reset(self):
        """
        reset the environment
        :return: observation after reset
        """
        # === state ===
        self.state.reset()  # clean
        self.step_count = 0
        self.state.set_ego(self._gen_vehicle_fix())
        # add random other vehicles
        for i in range(2):
            self._add_random_vehicle()
        self.action_space = [-1, 0, 1]

        # === pygame ===
        self.render()

        return self.get_observation(self.state.vehicles[0])

    def _get_random_vehicle(self):
        # generate a vehicle
        vehicle = None
        count = 0
        # try up to 5 times
        while not vehicle and count < 5:
            route = random.choice(self.road_map.routes)
            # route = self.road_map.routes[8]
            v = random.randrange(2, route.seg1.max_speed)
            vehicle = Vehicle(route, v, "other.png")
            if vehicle.collide(self.state.vehicles):
                vehicle = None
            count = count + 1
        # add to state
        return vehicle

    def _gen_vehicle_fix(self):
        return Vehicle(self.road_map.routes[5], 2, "ego.png")

    def render(self):
        """
        show graphic image of simulator
        """
        # erase window
        self.background = self.win.fill(Color.white, self.background)
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
        pg.time.wait(17)

    def get_observation(self, vehicle=None, v_id=None):
        """
        get Observation from current state based on a vehicle
        :param vehicle:
        :param v_id:
        :return:
        """
        other_vehicles = self.state.vehicles.copy()
        if not vehicle:
            if not v_id:
                raise Exception("must give a vehicle or a vehicle id")
            vehicle = self.state.get_vehicle_by_id(v_id)
        other_vehicles.remove(vehicle)
        others = [other.get_observation() for other in other_vehicles]
        return Observation(vehicle, others, self.road_map)

    def get_others_actions(self):
        actions = []
        for v_id, agent in self.state.agents.items():
            actions.append(agent.get_action(self.get_observation(v_id=v_id)))
        return actions

    def _add_random_vehicle(self):
        other = self._get_random_vehicle()
        if other:
            self.state.add_others(other, TTC())

    def done(self):
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
        return done

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
        # add random vehicle with rate
        if random.random() < 0:
            self._add_random_vehicle()
        # get the action for other vehicles
        actions = self.get_others_actions()
        actions.insert(0, action)
        self.state.step(actions, self.dt)
        self.step_count += 1

        observation = self.get_observation(self.state.vehicles[0])

        return observation, self.done(), self.step_count

    def __del__(self):
        pass


class InfiniteEnv(Env):

    def __init__(self, params):
        super().__init__(params)

    def step(self, action=None):
        if random.random() < 0.05:
            self._add_random_vehicle()
        # get the action for other vehicles
        actions = self.get_others_actions()
        self.state.step(actions, self.dt)
        self.step_count += 1

        observation = self.get_observation(self.state.vehicles[0])

        return observation, self.done(), self.step_count

    def done(self):
        ego = self.state.vehicles[0]
        if not ego.exist:
            self.state.vehicles.remove(ego)
        if not self.state:
            return 1
        elif max_steps < self.step_count:
            return 2
        else:
            return 0
