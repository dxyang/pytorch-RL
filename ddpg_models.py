import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

class Actor_Policy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size_1 = 400, hidden_size_2= 300, unif_range=3e-3):
        super(Actor_Policy, self).__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(num_inputs, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, num_actions)
        self.init_weights(unif_range)
        self.tanh = nn.Tanh()

    def fan_in(self, size):
        bound = 1.0 / np.sqrt(size[0])
        return torch.Tensor(size).uniform_(-bound, bound)

    def init_weights(self, unif_range):
        self.fc1.weight.data = self.fan_in(self.fc1.weight.data.size())
        self.fc2.weight.data = self.fan_in(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-unif_range, unif_range)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class Critic_Value(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size_1 = 400, hidden_size_2= 300, unif_range=3e-3):
        super(Critic_Value, self).__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(num_inputs, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1 + num_actions, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, 1)

        self.init_weights(unif_range)

    def fan_in(self, size):
        bound = 1.0 / np.sqrt(size[0])
        return torch.Tensor(size).uniform_(-bound, bound)

    def init_weights(self, unif_range):
        self.fc1.weight.data = self.fan_in(self.fc1.weight.data.size())
        self.fc2.weight.data = self.fan_in(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-unif_range, unif_range)

    def forward(self, inputs):
        state, actions = inputs
        x = F.relu(self.fc1(state))
        x = torch.cat((x, actions), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
