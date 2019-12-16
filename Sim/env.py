#!/usr/bin/env python3

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
from agent import *
from Sim.model import *
from Sim.roadmap import *
import random

# max step number
max_steps = 300


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


class Env(object):
    r"""The main Sim class.

        The main API methods that users of this class need to know are:

            step
            reset
            render
            close

    """

    def __init__(self, params):
        self.road_map = None
        self.vehicles = {}
        self.vehicles_count = 0
        self.state = State()
        self.action_space = range(-5, 3, 1)
        self.dt = 0.1  # s, interval
        self.steps = 0
        self.need_render = False

        # for render
        self.size_x = (params.x_max - params.x_min) * params.scale
        self.size_y = (params.y_max - params.y_min) * params.scale
        self.win = None
        self.background = None
        self.font = None

    def reset(self):
        """
        reset the environment
        :return: observation after reset
        """
        # clear
        self.vehicles.clear()
        self.steps = 0
        self.vehicles_count = 0

        # set initial env
        self.vehicles[self.vehicles_count] = Vehicle(self.road_map.routes[5], v=4, v_id=self.vehicles_count,
                                                     image_name="ego.png", max_speed=5, max_acc=1, min_acc=-5)
        self.vehicles_count += 1

        self.vehicles[self.vehicles_count] = Vehicle(self.road_map.routes[10], v=3, v_id=self.vehicles_count,
                                                     agent=TTC(), image_name="other.png", max_speed=4, max_acc=1,
                                                     min_acc=-5)
        self.vehicles_count += 1
        self.vehicles[self.vehicles_count] = Vehicle(self.road_map.routes[2], v=5, v_id=self.vehicles_count,
                                                     agent=TTC(), image_name="other.png", max_speed=6, max_acc=2,
                                                     min_acc=-5)
        self.vehicles_count += 1

        return self._get_observation(0)

    def step(self, action):
        """
        taken an action and return the simulation results
        :param action: action needs to be taken
        :return:
        Observation: new observation after taking an action
        reward: R(s,a)
        done: the simulation is finished
        info: debug message
        """
        self._transition(self.action_space[action])
        self.steps += 1
        done = self._is_done()
        observation = self._get_observation(0) if done == 0 else None
        reward = self._get_reward(done)

        return observation, reward, done, self.steps

    def render(self):
        """
        show graphic image of simulator
        """
        if not self.need_render:
            self.need_render = True
            pg.init()
            self.win = pg.display.set_mode((self.size_x, self.size_y))
            pg.display.set_caption("Autonomous driving simulator")
            self.background = (0, 0, self.size_x, self.size_y)
            self.font = pg.font.SysFont("arial", 16)
        self.background = self.win.fill(Color.white, self.background)
        self.road_map.render(self.win)
        for vehicle in self.vehicles.values():
            vehicle.render(self.win)
        # quit event
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                exit(0)
        pg.display.update()
        pg.time.wait(17)

    def close(self):
        if self.need_render:
            pg.quit()

    def _get_random_vehicle(self):
        # generate a vehicle
        vehicle = None
        count = 0
        # try up to 5 times
        while not vehicle and count < 5:
            route = random.choice(self.road_map.routes)
            # route = self.road_map.routes[8]
            v = random.randrange(2, route.seg1.max_speed)
            vehicle = Vehicle(route, v, v_id=self.vehicles_count, agent=TTC(), image_name="other.png")
            if self._collide(vehicle.id):
                vehicle = None
            count = count + 1
        # add to state
        return vehicle

    def _get_observation(self, v_id):
        vehicles = {v_id: self.vehicles[v_id]}
        for key, vehicle in self.vehicles.items():
            if v_id != key and self._is_observable(v_id, key):
                vehicles[key] = vehicle
        return Observation(v_id, self.vehicles, self.vehicles_count)

    def _is_observable(self, v_id, o_id):
        if self.vehicles[o_id].exist:
            return True
        return False

    def _add_random_vehicle(self):
        other = self._get_random_vehicle()
        if other:
            self.vehicles[self.vehicles_count] = other
            self.vehicles_count += 1

    def _collide(self, v_id):
        others = []
        for o_id, vehicle in self.vehicles.items():
            if o_id != v_id and vehicle.exist:
                others.append(vehicle)
        return self.vehicles[v_id].collide(others)

    def _is_done(self):
        if self.vehicles[0].exist:
            done = -1 if self._collide(0) else 0
        else:
            done = 1  # out of map
        if not done and self.steps > max_steps:
            done = -2  # overtime
        return done

    def _transition(self, action):
        self.vehicles[0].step(action, self.dt)
        for v_id, vehicle in self.vehicles.items():
            if v_id != 0 and vehicle.exist:
                o = self._get_observation(v_id)
                a = vehicle.agent.get_action(o)
                vehicle.step(a, self.dt)

    def _get_reward(self, done):
        if done == 1:
            return 20000
        ego = self.vehicles[0]
        ra = -100 * ego.action ** 2
        rv = -400 * (ego.v - ego.get_max_speed()) ** 2
        rc = -20000 if done == -1 else 0

        return ra + rv + rc


def make(name):
    if name == "t":
        env = Env(InterParam)
        env.road_map = _gen_t_inter(InterParam)
    elif name == "full":
        env = Env(InterParam)
        env.road_map = _gen_full_inter(InterParam)
    else:
        raise Exception("no such environment")
    return env
