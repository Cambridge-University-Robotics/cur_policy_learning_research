import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dm_control import suite, viewer
from tqdm import tqdm

from simulation.dm_control_cur.ddpg.ddpg_classes.ddpg_base import DDPGagent
from simulation.dm_control_cur.ddpg.ddpg_classes.utils import MemorySeq
from simulation.dm_control_cur.utility_classes.abstract_classes import AbstractSimulation


class Simulation(AbstractSimulation):
    def __init__(
            self,

            # Generic parameters for simulation
            load_model=False,
            plot=True,
            label=None,  # tag that is appended to file name for models and graphs
            num_episodes=50,  # number of simulation rounds before training session
            batch_size=128,  # number of past simulations to use for training
            duration=50,  # duration of simulation

            # Injecting environment
            env=None,
            name_model='cartpole',
            task='balance',
            dim_action=None,
            dim_obs=None,

            # Injecting agent
            model=DDPGagent,
            model_inputs=None,

            # Injecting agent (for backwards compatibility)
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-3,
            hidden_size=256,
            hidden_depth=1,
            gamma=0.99,
            tau=1e-2,
            memory=MemorySeq,

            # Miscellaneous
            random_state=np.random.RandomState(42),
            date_time=datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
    ):
        """
        All constants in caps, all objects in snake case
        Load everything, make directories etc
        """
        self.DURATION = duration
        self.BATCH_SIZE = batch_size
        self.NUM_EPISODES = num_episodes
        self.PLOT = plot
        self.MODELS_STR = 'models'
        self.DATA_STR = 'data'
        self.NAME_MODEL = name_model
        self.TASK = task
        self.LABEL = f'{label}_' if label is not None else ''
        self.MODEL_PATH = f'{self.MODELS_STR}/{self.LABEL}{name_model}_{task}'
        self.DATA_PATH = f'{self.DATA_STR}/{self.LABEL}{name_model}_{task}_{date_time}'
        env = env or suite
        self.env = env.load(name_model, task, task_kwargs={'random': random_state})
        action_spec = self.env.action_spec()
        dim_action = dim_action or action_spec.shape[0]
        dim_obs = dim_obs or sum(tuple(map(lambda x: int(np.prod(x.shape)), self.env.observation_spec().values())))
        self.MODEL_INPUTS = {
            "num_states": dim_obs,
            "num_actions": dim_action,
            "action_low": action_spec.minimum,
            "action_high": action_spec.maximum,
            "actor_learning_rate": actor_learning_rate,
            "critic_learning_rate": critic_learning_rate,
            "hidden_depth": hidden_depth,
            "hidden_size": hidden_size,
            "gamma": gamma,
            "tau": tau,
            "memory": memory,
        }
        self.MODEL_INPUTS.update(model_inputs or {})
        self.agent = model(**self.MODEL_INPUTS)
        Path(self.MODELS_STR).mkdir(parents=True, exist_ok=True)
        Path(self.DATA_STR).mkdir(parents=True, exist_ok=True)
        if load_model: self.agent.load(self.MODEL_PATH)

    def get_action(self, state, t):
        return self.agent.get_action(state, t)

    def modify_obs(self, obs):
        """default dm_control env obs parsing"""
        x = np.array([])
        for _, v in obs.items():
            x = np.append(x, v)
        return x

    def modify_action(self, action, state, t):
        return action

    def train(self):
        rewards = []
        avg_rewards = []

        tqdm_range = tqdm(range(self.NUM_EPISODES))
        for episode in tqdm_range:
            time_step = self.env.reset()
            state = self.modify_obs(time_step.observation)
            episode_reward = 0

            for t in range(self.DURATION):
                action = self.agent.get_action(state, t=t)
                print(f'my action is: {action}-------------------')
                action_modified = self.modify_action(action, state, t)
                try:
                    time_step_2 = self.env.step(action_modified)
                except Exception as e:
                    print(e)
                    break
                state_2 = self.modify_obs(time_step_2.observation)
                self.agent.push(state, action, time_step_2.reward, state_2, -1)
                state = state_2
                self.agent.update(self.BATCH_SIZE)
                episode_reward += time_step_2.reward

            rewards.append(episode_reward)
            avg_reward = np.mean(rewards[-10:])
            avg_rewards.append(avg_reward)

            desc = f"episode: {episode}, " \
                   f"reward: {np.round(episode_reward, decimals=2)}, " \
                   f"average_reward: {avg_reward}"
            tqdm_range.set_postfix_str(s=desc)

            self.agent.save(self.MODEL_PATH)
            if self.PLOT:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(rewards)
                ax.plot(avg_rewards)
                ax.set_xlabel('Episode')
                ax.set_ylabel('Reward')
                # plt.ylim([0, self.DURATION])
                plt.xlim([0, self.NUM_EPISODES])
                fig.savefig(self.DATA_PATH)
                plt.close(fig)
                with open(f'{self.DATA_PATH}.json', "w") as fp:
                    obj = {
                        'name_model': self.NAME_MODEL,
                        'task': self.TASK,
                        'label': self.LABEL,
                        'num_episodes': self.NUM_EPISODES,
                        'batch_size': self.BATCH_SIZE,
                        'duration': self.DURATION,
                        'rewards': rewards,
                        'avg_rewards': avg_rewards,
                    }
                    obj = dict(obj, **self.MODEL_INPUTS)

                    def is_jsonable(x):
                        try:
                            json.dumps(x)
                            return True
                        except (TypeError, OverflowError):
                            return False

                    obj = {k: v for k, v in obj.items() if is_jsonable(v)}
                    json.dump(obj=obj, fp=fp)

    def show_simulation(self):
        t = -1

        def policy(time_step):
            nonlocal t
            t += 1
            state = self.modify_obs(time_step.observation)
            
            action = self.agent.get_action(state, t)
            # print(action, t)
            #print(time_step.observation, state)
            action_modified = self.modify_action(action, state, t)
            #print(type(action_modified), type(action))
            return action_modified

        def policy_zero(time_step):
            state = self.modify_obs(time_step.observation)
            
            action = self.agent.get_action(state, t)
            state_zero = [0, 0, 0, -1 ,0]
            state_zero_modify = np.array(state_zero)
            state_zero_modify = state_zero_modify.astype(np.float64)
            print(state_zero_modify)
            return state_zero_modify

        viewer.launch(self.env, policy=policy)
