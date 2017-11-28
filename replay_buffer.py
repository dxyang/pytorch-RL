from collections import deque
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def size(self):
        return self.buffer_size

    def current_count(self):
        return self.count

    def reset(self):
        self.buffer = deque()
        self.count = 0

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            
    def sample(self, n_samples):
        if n_samples > self.count:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, n_samples)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        for experience in batch:
            state_batch.append(experience[0])
            action_batch.append(experience[1])
            reward_batch.append(experience[2])
            next_state_batch.append(experience[3])
            done_batch.append(experience[4])

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
