import numpy as np
from enum import Enum
import random
import pygame

max_acc = 5


class Env:

    def __init__(self):
        """
        init simulation environment. Including state, action spaces, observation spaces
        """
        self.state = None
        self.actions = []
        self.interval = 0.1
        self.time = 0
        self.max_time = 10
        self.reward = 0.0
        self.done = False
        self.is_render = False
        self.screen = None

    def reset(self):
        """
        reset the environment
        :return: observation after reset
        """
        self.state = State()
        self._compute_action()
        self.time = 0
        self.reward = 0.0
        self.done = False
        self.is_render = False
        return self.state

    def render(self):
        """
        show graphic image of simulator
        """
        if not self.is_render:
            pygame.init()
            self.screen = pygame.display.set_mode((400, 400), 0, 32)
        ego_loc, car_loc = self._transform()
        ego = pygame.Rect(ego_loc)
        car = pygame.Rect(car_loc)
        self.is_render = True

    def _transition(self, action):
        # state change of ego vehicle
        self.state.ego.v += self.interval * action
        self.state.ego.pos += (self.interval ** 2) * action / 2
        # state change of other vehicle
        self.state.car.pos += self.state.car.v * self.interval
        # compute reward before update acceleration
        self.reward = self._get_reward__()
        self.state.ego.a = action

    def _transform(self):
        ego = self.state.ego
        car = self.state.car
        ego_loc = ((-ego.width / 2, ego.pos), (ego.width / 2, ego.pos - ego.length))
        car_loc = ((car.pos - car.length, car.width / 2), (car.pos, -car.width / 2))
        return ego_loc, car_loc

    def _compute_action(self):
        action = Action()
        state = self.state
        self.actions.clear()
        # self.actions.append(-1)
        self.actions.append(0)
        self.actions.append(1)
        # self.actions.append(action.keep_speed(state.ego.v))
        # self.actions.append(action.keep_distance(state.ego, state.car))
        # self.actions[2] = action.keep_distance(state.ego, state.cars[0])
        # self.actions[3] = action.keep_distance(state.ego, state.cars[1])
        # self.actions[4] = action.keep_distance(state.ego, state.cars[2])
        # self.actions[5] = action.keep_distance(state.ego, state.cars[3])

    def _get_reward__(self):
        self.done = True
        if self._success():
            return 1.0 - self.time / self.max_time
        if self._failure():
            return -2.0
        if self._time_out():
            return -0.1
        self.done = False
        return 0

    def _success(self):
        ego = self.state.ego
        # success when the ego vehicle pass the intersection
        if ego.pos - ego.length > ego.delta:
            return True
        return False

    def _failure(self):
        # collide
        if self.state.ego.collide(self.state.car):
            return True
        return False

    def _time_out(self):
        if self.time > self.max_time:
            return True
        return False

    # def _done(self):
    #     return self._success() or not self._failure()

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
        self._transition(action)
        return self.state, self.reward, self.done

    def __del__(self):
        pass


class Intent(Enum):
    take_way = 1
    give_way = 2
    cautious = 3


class Vehicle:
    """
    a single vehicle's state and its attribute
    Each state has the following attribute:
        pos: position of the vehicle, defined as distance to the center of the intersection.
            Negative value means the vehicle is move toward the intersection.
        v: speed
        a: acceleration
        max_speed: speed limit of a road
        delta: distance from stop line to the center of the intersection
        intent: intention, only the other vehicles have intention as a hidden attribute
    Attributes are: shape of the vehicle, action that can be take
    """

    def __init__(self, pos=-10, v=0, a=0, max_speed=30, delta=4, intent=Intent.cautious):
        self.pos = pos
        self.v = v
        self.a = a
        self.delta = delta
        self.intent = intent
        self.max_speed = max_speed
        self.length = 2
        self.width = 1
        self.action = None

    def collide(self, other):
        # other vehicle's body is in collision area
        if other.pos > - self.width / 2 and other.pos - other.length < self.width / 2:
            # detect whether the ego vehicle is in
            if self.pos > - other.width / 2 and self.pos - self.length < other.width / 2:
                return True
        return False


class State(object):
    """
    State represents the whole environment including four vehicles' state and the ego vehicle's state.
    """

    def __init__(self):
        self.ego = Vehicle(-random.randint(4, 10))
        self.car = Vehicle(-random.randint(4, 10), 10)
        # self.cars = []
        # self.cars_count = 0  # random.randint(1, 4)
        # for i in range(self.cars_count):
        #     self.cars[i] = Vehicle(random.randint(4, 10), 0, 0, 4, random.randint(1, 3))
        #     self.cars[i].action =


class Observation:
    """
    Observation is the visible part that the ego vehicle can observe from current state.
    In detail, observation contains the position, speed, acceleration and start of intersection of every vehicle
    and the predicted acceleration of ego vehicle in each action
    """

    def __init__(self, pos=10, v=0, a=0, delta=4):
        self.pos = pos
        self.v = v
        self.a = a
        self.delta = delta


class Observations:

    def __init__(self, state):
        self.state = state


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
        return min(self.K * (self.v_max - v), max_acc)

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
