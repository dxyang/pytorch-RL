import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import math

from utils import *
from ddpg_models import Actor_Policy, Critic_Value
from replay_buffer import ReplayBuffer

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class DDPG(object):
    def __init__(self,
                 observation_dim,
                 num_actions, 
                 batch_size, 
                 gamma,
                 d_epsilon,
                 update_rate,
                 is_train):
        self.observation_dim = observation_dim
        self.num_actions = num_actions

        self.actor = Actor_Policy(observation_dim, num_actions).type(dtype)
        self.actor_target = Actor_Policy(observation_dim, num_actions).type(dtype)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic_Value(observation_dim, num_actions).type(dtype)
        self.critic_target = Critic_Value(observation_dim, num_actions).type(dtype)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-2)

        self.replay_buffer = ReplayBuffer(1e6)

        self.ornstein_uhlenbeck = OrnsteinUhlenbeckProcess(theta=0.15, sigma=0.2)

        self.batch_size = batch_size
        self.update_rate = update_rate
        self.epsilon = 1
        self.d_epsilon = 1.0/d_epsilon
        self.is_train = is_train
        self.gamma = gamma

    def update_target(self, target, original, update_rate):
        for target_param, param in zip(target.parameters(), original.parameters()):
            target_param.data.copy_((1.0 - update_rate) * target_param.data + update_rate *param.data)

    def select_action(self, state):
        obs = Variable((torch.from_numpy(np.array([state])))).type(dtype)
        action = self.actor(obs).cpu().squeeze(dim=0).data.numpy()
        action = action + (self.is_train) * max(self.epsilon, 0) * self.ornstein_uhlenbeck.sample()
        action = np.clip(action, -1.0, 1.0)
        if (self.epsilon > 0):
            self.epsilon -= self.d_epsilon
        return action

    def reset(self):
        self.ornstein_uhlenbeck.reset_states()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update_model(self):
        if (self.replay_buffer.current_count() < self.batch_size):
            return

        state_batch, action_batch, reward_batch, \
        next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        state_batch = Variable(torch.from_numpy(np.array(state_batch))).type(dtype)
        action_batch = Variable(torch.from_numpy(np.array(action_batch))).type(dtype)
        reward_batch = Variable(torch.from_numpy(np.array(reward_batch))).type(dtype)
        next_state_batch = Variable(torch.from_numpy(np.array(next_state_batch))).type(dtype)
        done_mask = Variable(torch.from_numpy(1 - np.array([done_batch]).T.astype(int))).type(dtype)

        # -----
        # Compute Bellman error to update critic
        # -----
        # (a) Q(s', mu(s'|theta_mu_frozen) | theta_q_frozen)
        action_tp1_target = self.actor_target(next_state_batch)
        Q_target_tp1_values = self.critic_target([next_state_batch, action_tp1_target]).detach()

        # if current state is end of episode, then there is no next Q value
        Q_target_tp1_values = done_mask * Q_target_tp1_values 

        # (b) Q(s, a | theta_q)
        Q_values = self.critic((state_batch, action_batch))

        # r + gamma * (a) - (b)
        y_i = reward_batch + self.gamma * Q_target_tp1_values
        critic_loss = nn.MSELoss()
        error = critic_loss(Q_values, y_i)

        self.critic_optimizer.zero_grad()
        error.backward()
        self.critic_optimizer.step()

        # -----
        # Update actor using critic
        # -----
        predicted_actions = self.actor(state_batch)
        actor_loss = (- self.critic([state_batch, predicted_actions])).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -----
        # Update target networks
        # -----
        self.update_target(self.critic_target, self.critic, self.update_rate)
        self.update_target(self.actor_target, self.actor, self.update_rate)
