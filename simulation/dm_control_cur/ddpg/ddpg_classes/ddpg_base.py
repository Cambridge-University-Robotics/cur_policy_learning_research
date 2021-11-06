import torch.autograd
import torch.optim as optim

from .model import *
from .utils import *


class Agent(ABC):
    """
    Usage:
    1. initialise
    2. get_action (inputs and outputs are normed and denormed)
    3. push to add experience to memory (include arguments if necessary)
    4. update to train model
    """

    @abstractmethod
    def get_action(self, state, t):
        pass

    @abstractmethod
    def push(self, state, action, reward, next_state, done, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, batch_size):
        pass

    @abstractmethod
    def save(self, path_save):
        pass

    @abstractmethod
    def load(self, path_load):
        pass


class DDPGagent(Agent):
    def __init__(
            self,
            num_states,
            num_actions,
            action_low,
            action_high,
            hidden_size=256,
            hidden_depth=1,
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-3,
            gamma=0.99,  # reward decay
            tau=1e-2,  # fraction of weights which are copied to the target networks
            max_memory_size=50000,
            memory=MemorySeq,
            noise=OUNoise
    ):
        # Params
        self.num_states = num_states
        self.num_actions = num_actions
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.noise = noise(action_dim=num_actions, action_low=action_low, action_high=action_high)

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions, hidden_depth)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions, hidden_depth)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions, hidden_depth)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions,
                                    hidden_depth)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    # use on env output before passing to model
    def _denorm(self, action):
        act_k = (self.action_high - self.action_low) / 2.
        act_b = (self.action_high + self.action_low) / 2.
        return action * act_k + act_b

    # use on model output before passing to env
    def _norm(self, action):
        act_k_inv = 2. / (self.action_high - self.action_low)
        act_b = (self.action_high + self.action_low) / 2.
        return act_k_inv * (action - act_b)

    def get_action(self, state, t):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.actor(state)
        action = action.detach().numpy()[0]
        self.noise.reset() if t == 0 else None
        action = self.noise.get_action(action, t)
        action = self._denorm(action)
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Critic loss
        Qvals = self.critic(states, actions)
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def push(self, state, action, reward, next_state, done, *args, **kwargs):
        action = self._norm(action)
        self.memory.push(state, action, reward, next_state, done, *args, **kwargs)

    def save(self, path_save):
        torch.save(self.actor, f'{path_save}_actor.pt')
        torch.save(self.actor_target, f'{path_save}_actor_target.pt')
        torch.save(self.critic, f'{path_save}_critic.pt')
        torch.save(self.critic_target, f'{path_save}_critic_target.pt')

    def load(self, path_load):
        self.actor = torch.load(f'{path_load}_actor.pt')
        self.actor_target = torch.load(f'{path_load}_actor_target.pt')
        self.critic = torch.load(f'{path_load}_critic.pt')
        self.critic_target = torch.load(f'{path_load}_critic_target.pt')
