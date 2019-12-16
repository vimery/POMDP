import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent:
    """
    Agent: basic class represents a planner
    """

    def __init__(self):
        pass

    def get_action(self, state):
        pass


class TTC(Agent):
    """
    Time to collision, a model that describes how a driver will do with assumption of constant speed
    """

    def __init__(self):
        super().__init__()
        self.pre = 4
        self.ego = None

    def collide_predict(self, observation):
        ego = observation.ego
        return False

    def get_action(self, observation):
        return 0


class Constant(Agent):

    def __init__(self):
        super().__init__()
        self.acc = 0

    def get_action(self, state):
        return self.acc


class DQNAgent(Agent):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def get_action(self, ob):
        x = torch.tensor(ob.get_array())
        x.view((-1, 1))
        return 0


class DQN(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
