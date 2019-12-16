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
        self.vehicles = []
        self.state = State()
        self.action_space = []
        self.dt = 0.1  # s, interval
        self.steps = 0
        self.need_render = False

        # for render
        self.size_x = (params.x_max - params.x_min) * params.scale
        self.size_y = (params.y_max - params.y_min) * params.scale
        self.win = None
        self.background = None
        self.font = None

        self.reset()

    def reset(self):
        """
        reset the environment
        :return: observation after reset
        """
        # clear
        self.vehicles.clear()
        self.steps = 0

        # set initial env
        self.vehicles.append(
            Vehicle(self.road_map.routes[5], v=4, image_name="ego.png", max_speed=5, max_acc=1, min_acc=-5)
        )
        self.vehicles.append(
            Vehicle(self.road_map.routes[10], v=3, image_name="other.png", max_speed=4, max_acc=1, min_acc=-5)
        )
        self.vehicles.append(
            Vehicle(self.road_map.routes[2], v=5, image_name="other.png", max_speed=6, max_acc=2, min_acc=-5)
        )
        self.action_space = [-2, 0,  2]

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

    def render(self):
        """
        show graphic image of simulator
        """
        if not self.need_render:
            self.need_render = True
            pg.init()
            self.win = pg.display.set_mode((self.size_x, self.size_y))
            pg.display.set_caption("Autonomous driving simulator")
            self.background = self.win.fill(Color.white, (0, 0, self.size_x, self.size_y))
            self.font = pg.font.SysFont("arial", 16)
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
        if not done and self.steps > max_steps:
            done = -2  # overtime
        return done

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
        # get the action for other vehicles
        actions = self.get_others_actions()
        actions.insert(0, action)
        self.state.step(actions, self.dt)
        self.steps += 1

        observation = self.get_observation(self.state.vehicles[0])

        return observation, self.done(), self.steps

    def __del__(self):
        pass

    def close(self):
        if self.need_render:
            pg.quit()


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
