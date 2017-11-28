import torch
from torch.autograd import Variable
import torch.nn.utils as utils
import gym
from gym import wrappers
import numpy as np
import os

from ddpg import DDPG
from normalized_actions import *
from utils import *
from logger import Logger

NUM_EPISODES = 5000
NUM_STEPS = 500
GAMMA = 0.99
CKPT_FREQ = 100
BATCH_SIZE = 64
UPDATE_RATE = 0.001
RENDER_VIDEO = 100
SAVE_VIDEO = 25

# Gym things
env_name = 'HalfCheetah-v1' #'Pendulum-v0' #'HalfCheetah-v1' #'MountainCarContinuous-v0' #'Hopper-v1' #'InvertedPendulum-v1'
env = NormalizedActions(gym.make(env_name))
monitor_dir = 'tmp/' + env_name
if not os.path.exists(monitor_dir):
    os.mkdir(monitor_dir)
env = wrappers.Monitor(env, monitor_dir, force=True, video_callable=lambda episode_id: episode_id%SAVE_VIDEO==0)
env._max_episodes_steps = NUM_STEPS

# Learning agent
agent = DDPG(observation_dim=env.observation_space.shape[0],
             num_actions=env.action_space.shape[0], 
             batch_size=BATCH_SIZE, 
             gamma=GAMMA,
             d_epsilon=50000,
             update_rate=UPDATE_RATE,
             is_train=True)

# Set the logger
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
logger = Logger(log_dir)


num_steps = 0
episode_rewards = []
for i_episode in range(NUM_EPISODES):
    state = env.reset()

    rewards = []

    #for i in range(NUM_STEPS):
    while True:
        if (i_episode % RENDER_VIDEO == 0):
            env.render()
        action = agent.select_action(state)

        next_state, reward, done, _ = env.step(action[0])

        agent.store_experience(state, action, [reward], next_state, done)

        rewards.append(reward)
        state = next_state

        agent.update_model()
        agent.reset()

        num_steps += 1

        if done:
            break

    episode_reward = np.sum(rewards)
    episode_rewards.append(episode_reward)
    if i_episode % CKPT_FREQ == 0:
        torch.save(agent.critic_target.state_dict(), os.path.join(monitor_dir, 'critic-' + str(i_episode) + '.pkl'))
        torch.save(agent.actor_target.state_dict(), os.path.join(monitor_dir, 'actor-' + str(i_episode) + '.pkl'))

    if i_episode % 1 == 0:
        print("Episode: {}, i {}, reward: {}".format(i_episode, num_steps, episode_reward))


    #============ TensorBoard logging ============#
    if len(episode_rewards) >= 100:
        last_100_reward = np.mean(episode_rewards[-100:])
    else:
        last_100_reward = np.mean(episode_rewards)

    info = {
        'reward_per_episode': episode_reward,
        'mean_reward_last_100_episodes': last_100_reward,
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, i_episode+1)

    info = {
        'reward_per_episode_over_iters': episode_reward,
        'mean_reward_last_100_episodes_over_iters': last_100_reward,
        'num_episodes_per_steps': i_episode,
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, num_steps)

env.close()


