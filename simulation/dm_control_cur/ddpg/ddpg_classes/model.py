import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer_count):
        super(Critic, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layer_count)])
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear_in(x))
        for f in self.hidden:
            x = F.relu(f(x))
        x = self.linear_out(x)
        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layer_count):
        super(Actor, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layer_count)])
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear_in(state))
        for f in self.hidden:
            x = F.relu(f(x))
        x = torch.tanh(self.linear_out(x))
        return x
