import copy
import time
import numpy as np
import torch
from torch.nn import MSELoss

from simulation.dm_control_cur.utility_classes.abstract_classes import Agent
from structures.networks import Actor, Critic
from structures.replay_memory import ReplayMemory
from structures.transition import Transition


class TD3Agent(Agent):
    def __init__(self, memory_size: int, states_dim: int, actions_dim: int, action_min: float,
                 action_max: float, noise: float, target_noise: float, noise_clip: float, q_polyak: float,
                 policy_polyak: float, gamma: float,
                 q_hidden_units: int, policy_hidden_units: int, batch_size: int, q_learning_rate: float,
                 pi_learning_rate: float, device,
                 policy_update_delay: int = 2):

        self.q_learning_rate = q_learning_rate
        self.pi_learning_rate = pi_learning_rate
        self.policy_update_delay = policy_update_delay
        self.q_polyak = q_polyak
        self.policy_polyak = policy_polyak

        """Initializing Q networks"""
        self.Q1 = Critic(states_dim=states_dim, hidden_units=q_hidden_units, action_dim=actions_dim).to(device)
        self.Q1_optimizer = torch.optim.Adam(self.Q1.parameters(), lr=self.q_learning_rate)
        self.Q2 = Critic(states_dim=states_dim, hidden_units=q_hidden_units, action_dim=actions_dim).to(device)
        self.Q2_optimizer = torch.optim.Adam(self.Q2.parameters(), lr=self.q_learning_rate)
        self.target_Q1 = copy.deepcopy(self.Q1)
        self.target_Q2 = copy.deepcopy(self.Q2)

        """Now policy networks"""
        self.policy = Actor(action_dim=actions_dim, states_dim=states_dim, hidden_units=q_hidden_units, max_action=action_max).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.pi_learning_rate)
        self.target_policy = copy.deepcopy(self.policy)

        self.loss_fn = MSELoss()

        self.memory = ReplayMemory(capacity=memory_size, states_dim=states_dim, actions_dim=actions_dim, device=device)
        self.device = device

        """noise hparams"""
        self.noise = noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.batch_size = batch_size
        self.gamma = gamma
        self.states_dim = states_dim
        self.action_dim = actions_dim
        self.action_min = action_min
        self.action_max = action_max

        self.update_iterator = 0

        """for inspection purposes"""
        self.q_upd_t = []
        self.pi_upd_t = []
        self.sampling_t = []
        self.rewriting_nets_t = []

    def get_action(self, state: np.ndarray, noise: float = None):
        policy_input = torch.tensor(state, dtype=torch.float).to(self.device)
        action = self.policy(policy_input)

        action = action.cpu().detach().numpy()
        if noise is None:
            noise = self.noise
        action = np.add(action, np.random.normal(0, noise, size=(1, self.action_dim)))

        return np.clip(action, a_min=self.action_min, a_max=self.action_max)

    def update(self, batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size

        start = time.time()
        batch = self.memory.sample(batch_size)
        self.sampling_t.append(time.time() - start)

        start = time.time()
        self._update_Q(batch)
        self.q_upd_t.append(time.time() - start)

        self.update_iterator += 1
        if self.update_iterator % self.policy_update_delay == 0:
            start = time.time()
            self._update_policy(batch)
            self.pi_upd_t.append(time.time() - start)

            """update target networks"""
            start = time.time()
            with torch.no_grad():
                for param, targ_param in zip(self.Q1.parameters(), self.target_Q1.parameters()):
                    targ_param.copy_((1-self.q_polyak) * param.data + self.q_polyak*targ_param.data)

                for param, targ_param in zip(self.Q2.parameters(), self.target_Q2.parameters()):
                    targ_param.copy_((1-self.q_polyak) * param.data + self.q_polyak*targ_param.data)

                for param, targ_param in zip(self.policy.parameters(), self.target_policy.parameters()):
                    targ_param.copy_((1-self.policy_polyak) * param.data + self.policy_polyak*targ_param.data)
            self.rewriting_nets_t.append(time.time() - start)

    def _calc_targets(self, batch):
        with torch.no_grad():
            next_actions = self.target_policy(batch['next_states'])
            noise = torch.clip(torch.normal(mean=torch.zeros(size=next_actions.size()),
                                            std=torch.full(size=next_actions.size(), fill_value=self.target_noise)),
                               min=-self.noise_clip, max=self.noise_clip).to(self.device)
            next_actions = torch.clip(next_actions + noise, min=self.action_min, max=self.action_max)

            y1 = self.target_Q1(state=batch['next_states'], action=next_actions)
            y2 = self.target_Q2(state=batch['next_states'], action=next_actions)

            return batch['rewards'] + (1 - batch['terminated']) * self.gamma*torch.minimum(y1, y2)

    def _update_Q(self, batch):
        targets = self._calc_targets(batch)
        Q1_loss = self.loss_fn(targets, self.Q1(state=batch['states'], action=batch['actions']))
        Q2_loss = self.loss_fn(targets, self.Q2(state=batch['states'], action=batch['actions']))

        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.Q1_optimizer.step()

        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.Q2_optimizer.step()

    def _update_policy(self, batch):
        mean_expected_reward = -self.Q1(state=batch['states'], action=self.policy(batch['states'])).mean()

        self.policy_optimizer.zero_grad()
        mean_expected_reward.backward()
        self.policy_optimizer.step()

    def push(self, state, action, reward, next_state, done, *args, **kwargs):
        self.memory.push(Transition(state, action, next_state, reward, done))

    def save(self, path_save):
        pass

    def load(self, path_load):
        pass
