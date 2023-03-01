import numpy as np
import torch
from transition import Transition


class ReplayMemory(object):

    def __init__(self, capacity: int, states_dim: int, actions_dim: int, device):
        self.device = device
        self.capacity = capacity

        self.states = np.zeros((capacity, states_dim))
        self.actions = np.zeros((capacity, actions_dim))
        self.next_states = np.zeros((capacity, states_dim))
        self.rewards = np.zeros((capacity, 1))
        self.terminated = np.zeros((capacity, 1))

        self.size = 0
        self.ptr = 0

    def push(self, transition: Transition):
        """
        Saves transition in preprocessed form. Every value is a tensor and appropriate tensors are normalized. So
        transition from memory is ready to feed into the network.
        """

        self.states[self.ptr] = transition.state.reshape(1, -1)
        self.actions[self.ptr] = transition.action.reshape(1, -1)
        self.next_states[self.ptr] = transition.next_state.reshape(1, -1)
        self.rewards[self.ptr] = transition.reward.reshape(1, -1)
        self.terminated[self.ptr] = np.reshape(1 if transition.terminated else 0, (1, ))

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = max(self.size, self.ptr)

    def sample(self, batch_size: int) -> dict:
        """

        :param batch_size:
        :return: batch in form of dictionary with tensors(every row of tensor correspond to particular transition)
        """
        indices = np.random.randint(0, self.size, size=min(batch_size, self.size))

        return {'states': torch.tensor(self.states[indices], dtype=torch.float).to(self.device),
                'actions': torch.tensor(self.actions[indices], dtype=torch.float).to(self.device),
                'next_states': torch.tensor(self.next_states[indices], dtype=torch.float).to(self.device),
                'rewards': torch.tensor(self.rewards[indices], dtype=torch.float).to(self.device),
                'terminated': torch.tensor(self.terminated[indices], dtype=torch.float).to(self.device)}

    def __len__(self):
        return self.size
