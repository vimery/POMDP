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
                    if o_x and collide_detection(e_x, e_y, o_x, o_y, ego.radius + ego.safe_distance,
                                                 vehicle.radius + vehicle.safe_distance):
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
    eps_start = 1
    eps_end = 0.1
    eps_decay = 1000
    batch_size = 256
    gamma = 0.99
    lr = 1e-3
    target_update = 100
    memory_capacity = 10000

    def __init__(self, n_features, n_actions, prioritized=True):
        super().__init__()
        self.step = 0
        self.learn_count = 0
        self.device = torch.device("cuda:1" if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else "cpu")
        self.n_actions = n_actions
        self.policy_net = NN(n_features, n_actions).to(self.device)
        self.target_net = NN(n_features, n_actions).to(self.device)
        self.memory = ReplayMemory(self.memory_capacity)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.prioritized = prioritized

    def get_action(self, ob):  # ob: 1 * n_features
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.step / self.eps_decay)
        self.step += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(ob).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def get_action_without_exploration(self, ob):
        return self.policy_net(ob).max(1)[1].view(1, 1)

    def learn(self):
        if len(self.memory) < self.memory_capacity:
            return
        self.learn_count += 1
        if self.prioritized:
            indexes, transitions, weights = self.memory.sample(self.batch_size)
        else:
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
        if self.prioritized:
            td_errors = state_action_values - expected_state_action_values.unsqueeze(1)
            loss = torch.abs(td_errors) * torch.from_numpy(weights).to(device=self.device)
            loss = loss.mean()
            self.memory.update(indexes, np.abs(td_errors.detach().cpu().numpy()))
        else:
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target network, copying all weights and biases in DDQNP
        if self.learn_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))


class NN(nn.Module):

    def __init__(self, n_features, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(64, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(64, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    epsilon = 1e-4
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 1e-5
    size = 0

    def __init__(self, capacity, prioritized=True):

        self.tree = SumTree(capacity)
        self.prioritized = prioritized

    def push(self, *args):
        self.size += 1 if self.size < self.tree.capacity else 0
        max_p = self.tree.max()
        if max_p <= 0:
            max_p = 1
        self.tree.add(max_p, Transition(*args))
        # if self.size > self.tree.capacity:
        #     self.size %= self.tree.capacity

    def sample(self, batch_size):
        trans_list = np.empty(batch_size, dtype=object)
        indexes = np.empty(batch_size, dtype='int')
        weights = np.empty(batch_size, dtype='float32')
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i, rand in enumerate(np.random.uniform(0, self.tree.total_p, batch_size)):
            idx, p, data = self.tree.get(rand)
            trans_list[i] = data
            indexes[i] = idx
            weights[i] = np.power(self.tree.capacity * p / self.tree.total_p, -self.beta)
        return indexes, trans_list, weights / np.max(weights)

    def update(self, idx, err):
        ps = self._get_priority(err)
        for ti, p in zip(idx, ps):
            self.tree.update(ti, p)

    def _get_priority(self, td_error):
        return np.power(td_error + self.epsilon, self.alpha)

    def __len__(self):
        return self.size


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity
        self.index_leaf_start = capacity - 1

    def add(self, p, data):
        tree_idx = self.data_pointer + self.index_leaf_start
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            if left >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left]:
                    parent_idx = left
                else:
                    v -= self.tree[left]
                    parent_idx = right

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def max(self):
        return np.max(self.tree[-self.capacity:])

    @property
    def total_p(self):
        return self.tree[0]  # the root
