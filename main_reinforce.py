import torch
from torch.autograd import Variable
import torch.nn.utils as utils
import gym
from gym import wrappers
import numpy as np
import os

from reinforce import REINFORCE
from normalized_actions import *

NUM_EPISODES = 10000
NUM_STEPS = 500
GAMMA = 0.99
CKPT_FREQ = 100

env_name = 'HalfCheetah-v1' #'Hopper-v1'#'InvertedPendulum-v1'
env = NormalizedActions(gym.make(env_name))

agent = REINFORCE(128, env.observation_space.shape[0], env.action_space.shape[0])

dir = 'tmp'
if not os.path.exists(dir):
    os.mkdir(dir)

for i_episode in range(NUM_EPISODES):
    state = torch.Tensor([env.reset()])
    entropies = []
    log_probs = []
    rewards = []

    for t in range(NUM_STEPS):
        #if (i_episode % 100 == 0):
            #env.render()
        action, log_prob, entropy = agent.select_action(state, t)
        action = action.cpu()

        next_state, reward, done, _ = env.step(action.numpy()[0])

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = torch.Tensor([next_state])

        if done:
            break

    agent.update_parameters(rewards, log_probs, entropies, GAMMA)

    if i_episode % CKPT_FREQ == 0:
        torch.save(agent.model.state_dict(), os.path.join(dir, 'reinforce-' + str(i_episode) + '.pkl'))

    print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

env.close()



