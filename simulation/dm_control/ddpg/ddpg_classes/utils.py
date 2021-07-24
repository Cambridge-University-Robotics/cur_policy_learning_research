import numpy as np
import gym
from collections import deque
import random
from abc import ABC, abstractmethod


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_dim, action_low, action_high, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3,
                 decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.low = action_low
        self.high = action_high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)


class Memory(ABC):
    @abstractmethod
    def push(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass


class MemorySeq(Memory):
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class MemoryRank(Memory):
    def __init__(self, max_size, sort_period=1):
        self.max_size = max_size
        self.buffer = []
        self.sort_period = sort_period
        self.sort_ctr = sort_period
        self.weights = [1 / i for i in range(1, self.max_size + 1)]

    def push(self, state, action, reward, next_state, done, priority=0):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append((priority, experience))

    def sort(self):
        self.buffer.sort(key=lambda x: x[0], reverse=True)
        self.buffer = self.buffer[:self.max_size]

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        # increment counter and sort periodically
        if self.sort_ctr == self.sort_period:
            self.sort()
            self.sort_ctr = 0
        self.sort_ctr += 1

        # sampling is weighed using TD loss and rank-based representation
        batch = random.choices(population=self.buffer, weights=self.weights[:len(self.buffer)], k=batch_size)

        for _, experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
