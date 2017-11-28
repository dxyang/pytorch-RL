import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import math

from utils import *

# pdf of normal (P(X = x))
def normal(x, mu, sigma_sq):
    pi = Variable(torch.FloatTensor([math.pi])).expand_as(sigma_sq)
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b

class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_actions):
        super(Policy, self).__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2_mu = nn.Linear(hidden_size, num_actions)
        self.fc2_sigma_sq = nn.Linear(hidden_size, num_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        mu = self.fc2_mu(x)
        sigma_sq = self.fc2_sigma_sq(x)
        return mu, sigma_sq 

class REINFORCE(object):
    def __init__(self, 
                 hidden_size,
                 observation_dim,
                 num_actions):
     self.hidden_size = hidden_size
     self.observation_dim = observation_dim
     self.num_actions = num_actions

     self.model = Policy(hidden_size, observation_dim, num_actions)

     self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
     self.model.train()

    def select_action(self, state, t):
        mu, sigma_sq = self.model(Variable(state))
        sigma_sq = F.softplus(sigma_sq)

        eps = Variable(torch.randn(mu.size()))

        action = (mu + sigma_sq.sqrt() * eps).data
        prob = normal(action, mu, sigma_sq)

        # differential entropy for normal dist
        pi = Variable(torch.FloatTensor([math.pi])).expand_as(sigma_sq)
        e = Variable(torch.FloatTensor([math.e])).expand_as(sigma_sq)
        entropy = -0.5*((2*pi*e*sigma_sq).log())

        log_prob = prob.log()

        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            R_temp = Variable(R).expand_as(log_probs[i])
            loss -= (log_probs[i]*R_temp).sum() - (0.0001*entropies[i]).sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()

