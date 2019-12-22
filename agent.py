import math
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from Sim.tools import collide_detection


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
        self.pre = 3
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
                        return vehicle.route.priority <= ego.route.priority
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
        self.lr = 1e-4
        self.EPS_START = 1
        self.EPS_END = 0.05
        self.EPS_DECAY = 500
        self.batch_size = 256
        self.gamma = 0.999
        self.target_update = 100
        self.memory_capacity = 10000
        self.step = 0
        self.learn_count = 0
        self.device = torch.device("cuda:1" if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else "cpu")
        self.n_actions = n_actions
        self.policy_net = NN(n_features, n_actions).to(self.device)
        self.target_net = NN(n_features, n_actions).to(self.device)
        self.memory = ReplayMemory(self.memory_capacity)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def get_action(self, ob):  # ob: 1 * n_features
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.step / self.EPS_DECAY)
        self.step += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(ob).max(1)[1].view(1, 1)  #
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def get_action_without_exploration(self, ob):
        return self.policy_net(ob).max(1)[1].view(1, 1)

    def learn(self):
        if len(self.memory) < self.memory_capacity:
            return
        self.learn_count += 1
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # compute Q(s_t, a;\theta)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # compute Q(s_{t+1}, /argmax(Q(s_t, a;\theta);\theta^-)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        m_action_batch = self.policy_net(non_final_next_states).detach().max(1)[1].unsqueeze(1)
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).detach().max(1)[0]
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, m_action_batch).squeeze(1)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target network, copying all weights and biases in DQN
        if self.learn_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))


class DDPGAgent(Agent):

    def __init__(self, n_features, n_actions):
        super().__init__()
        self.device = torch.device("cuda:1" if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else "cpu")
        self.actor_eval = NN(n_features, n_actions).to(self.device)
        self.actor_target = NN(n_features, n_actions).to(self.device)
        self.critic_eval = NN(n_features, n_actions).to(self.device)
        self.critic_target = NN(n_features, n_actions).to(self.device)

        self.a_opt = optim.Adam(self.actor_eval.parameters(), lr=1e-4)
        self.c_opt = optim.Adam(self.critic_eval.parameters(), lr=1e-4)
        self.loss_td = nn.MSELoss()

        self.memory = ReplayMemory(10000)
        self.gamma = 0.999
        self.target_update = 10
        self.batch_size = 200

    def get_action(self, ob):
        return self.actor_eval(ob)[0].detach()

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

        # get actions
        actions = self.actor_eval(state_batch).gather(1, action_batch)
        # compute Q(s_t, a)
        state_action_values = self.critic_eval(state_batch).gather(1, action_batch)


class NN(nn.Module):

    def __init__(self, n_features, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
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
