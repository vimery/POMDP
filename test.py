import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

net = Net()
net = net.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

inputs = torch.tensor([[i, i ** 2, i ** 3] for i in range(10)], dtype=torch.float).to(device)
targets = torch.tensor([3 + 2 * i[1] + i[2] for i in inputs], dtype=torch.float).to(device)
targets = targets.view(-1, 1)
for i in range(5000):
    optimizer.zero_grad()

    output = net(inputs).to(device)
    loss = criterion(output, targets).to(device)
    loss.backward()
    optimizer.step()

    print(loss.item())
