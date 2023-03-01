import torch
import torch.nn as nn


class Actor(torch.nn.Module):
    def __init__(self,
                 states_dim: int,
                 hidden_units: int,
                 action_dim: int,
                 max_action: float = 1.0):
        super().__init__()
        self.l1 = nn.Linear(states_dim, hidden_units)
        self.l2 = nn.Linear(hidden_units, hidden_units)
        self.l3 = nn.Linear(hidden_units, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        l2_input = self.l1(state).relu()
        l3_input = self.l2(l2_input).relu()
        y = torch.tanh(self.l3(l3_input))*self.max_action

        return y


class Critic(torch.nn.Module):
    def __init__(self,
                 states_dim: int,
                 action_dim: int,
                 hidden_units: int):
        super().__init__()

        self.l1 = nn.Linear(states_dim + action_dim, hidden_units)
        self.l2 = nn.Linear(hidden_units, hidden_units)
        self.l3 = nn.Linear(hidden_units, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        critic_input = torch.cat(tensors=(state, action), dim=1)

        l2_input = self.l1(critic_input).relu()
        l3_input = self.l2(l2_input).relu()

        y = self.l3(l3_input)

        return y
