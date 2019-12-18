import math
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from Sim.tools import collide_detection

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200


class Agent:
    """
    Agent: basic class represents a planner
    """

    def __init__(self):
        pass

    def get_action(self, *args, **kwargs):
        pass


class TTC(Agent):
    """
    Time to collision, a model that describes how a driver will do with assumption of constant speed
    """

    def __init__(self, n_actions):
        super().__init__()
        self.pre = 2
        self.dt = 0.1
        self.n_actions = n_actions

    def collide_predict(self, ob):
        ego = ob.ego
        for dt in np.arange(0, self.pre, self.dt):
            e_x, e_y, _, _, _ = ego.forward(ego.action, dt)
            if not e_x:
                return False
            for v_id, vehicle in ob.vehicles.items():
                if v_id != ego.id:
                    o_x, o_y, _, _, _ = vehicle.forward(vehicle.action, dt)
                    if o_x and collide_detection(e_x, e_y, o_x, o_y, ego.radius, vehicle.radius):
                        return True
        return False

    def get_action(self, ob):
        if self.collide_predict(ob):
            return 0
        return self.n_actions - 1


class Constant(Agent):

    def __init__(self):
        super().__init__()
        self.acc = 0

    def get_action(self, state):
        return self.acc


class DQNAgent(Agent):

    def __init__(self, n_features, n_actions):
        super().__init__()
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else "cpu")
        self.n_actions = n_actions
        self.policy_net = DQN(n_features, n_actions).to(self.device)
        self.target_net = DQN(n_features, n_actions).to(self.device)
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.batch_size = 200
        self.gamma = 0.999
        self.target_update = 10

    def get_action(self, ob, step):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * step / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(ob).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # compute Q(s_{t+1}, a)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))


class DQN(nn.Module):

    def __init__(self, n_features, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 6)
        self.fc2 = nn.Linear(6, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
