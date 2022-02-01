from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from dm_control import suite, viewer
from tqdm import tqdm
from simulation.dm_control.ddpg.ddpg_classes.ddpg import DDPGagent
from simulation.dm_control.ddpg.ddpg_classes.utils import MemorySeq
from datetime import datetime
from pathlib import Path


class AbstractSimulation(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def show_simulation(self):
        pass

    @abstractmethod
    def get_action(self, state, t):
        pass


def parse(obs):
    x = np.array([])
    for _, v in obs.items():
        x = np.append(x, v)
    return x


class Simulation(AbstractSimulation):
    def __init__(
            self,
            load_model=False,
            plot=True,
            show_simulation=True,
            name_model='cartpole',
            task='balance',
            label=None,
            num_episodes=50,  # number of simulation rounds before training session
            batch_size=128,  # number of past simulations to use for training
            duration=50,  # duration of simulation

            random_state=np.random.RandomState(42),
            date_time=datetime.now().strftime("%d:%m:%Y-%H:%M:%S"),
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-3,
            gamma=0.99,
            tau=1e-2,
    ):
        # All constants in caps, all objects in snake case
        self.DURATION = duration
        self.BATCH_SIZE = batch_size
        self.NUM_EPISODES = num_episodes
        self.SHOW_SIMULATION = show_simulation
        self.PLOT = plot
        self.MODELS_STR = 'models'
        self.DATA_STR = 'data'
        self.NAME_MODEL = name_model
        self.TASK = task
        label = f'{label}_' if label is not None else ''
        self.MODEL_PATH = f'{self.MODELS_STR}/{label}{name_model}_{task}'
        self.DATA_PATH = f'{self.DATA_STR}/{label}{name_model}_{task}_{date_time}'
        self.env = suite.load(name_model, task, task_kwargs={'random': random_state})
        action_spec = self.env.action_spec()
        obs_spec = self.env.observation_spec()
        dim_action = action_spec.shape[0]
        dim_obs = sum(tuple(map(lambda x: int(np.prod(x.shape)), obs_spec.values())))
        self.agent = DDPGagent(
            num_states=dim_obs,
            num_actions=dim_action,
            action_low=action_spec.minimum,
            action_high=action_spec.maximum,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            tau=tau,
            memory=MemorySeq
        )
        Path(self.MODELS_STR).mkdir(parents=True, exist_ok=True)
        Path(self.DATA_STR).mkdir(parents=True, exist_ok=True)
        if load_model: self.agent.load(self.MODEL_PATH)

    def get_action(self, state, t):
        return self.agent.get_action(state, t)

    def train(self):
        rewards = []
        avg_rewards = []

        tqdm_range = tqdm(range(self.NUM_EPISODES))
        for episode in tqdm_range:
            time_step = self.env.reset()
            state = parse(time_step.observation)
            episode_reward = 0

            for t in range(self.DURATION):
                action = self.agent.get_action(state, t=t)
                time_step_2 = self.env.step(action)
                state_2 = parse(time_step_2.observation)
                self.agent.push(state, action, time_step_2.reward, state_2, -1)
                state = state_2
                self.agent.update(self.BATCH_SIZE)
                episode_reward += time_step_2.reward

            rewards.append(episode_reward)
            avg_rewards.append(np.mean(rewards[-10:]))
            desc = f"episode: {episode}, " \
                   f"reward: {np.round(episode_reward, decimals=2)}, " \
                   f"average_reward: {np.mean(rewards[-10:])}"
            tqdm_range.set_postfix_str(s=desc)

        self.agent.save(self.MODEL_PATH)

        if self.PLOT:
            plt.plot(rewards)
            plt.plot(avg_rewards)
            plt.plot()
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig(self.DATA_PATH)
            plt.show()

    def show_simulation(self):
        t = -1

        def policy(time_step):
            nonlocal t
            t += 1
            state = parse(time_step.observation)
            action = self.agent.get_action(state, t)
            return action

        viewer.launch(self.env, policy=policy)