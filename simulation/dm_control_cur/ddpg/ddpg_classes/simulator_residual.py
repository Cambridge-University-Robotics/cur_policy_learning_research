import matplotlib.pyplot as plt
import numpy as np
from dm_control import viewer
from tqdm import tqdm

from ddpg_classes.simulator import Simulation, parse


class ResidualSimulation(Simulation):
    def __init__(
            self,
            train_controller=True,
            controller_num_episodes=50,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.controller = Simulation(
            load_model=not train_controller,
            label='controller',
            name_model=self.NAME_MODEL,
            task=self.TASK,
            num_episodes=controller_num_episodes,
            batch_size=self.BATCH_SIZE,
            duration=self.DURATION,
        )
        if train_controller:
            self.controller.train()

    def show_controller_simulation(self):
        self.controller.show_simulation()

    def train(self):
        rewards = []
        avg_rewards = []

        tqdm_range = tqdm(range(self.NUM_EPISODES))
        for episode in tqdm_range:
            time_step = self.env.reset()
            state = parse(time_step.observation)
            episode_reward = 0

            for t in range(self.DURATION):
                action_a = self.agent.get_action(state, t=t)
                action_c = self.controller.get_action(state, t=t)
                action = action_a + action_c
                time_step_2 = self.env.step(action)
                state_2 = parse(time_step_2.observation)
                self.agent.push(state, action_a, time_step_2.reward, state_2, -1)
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
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(rewards)
            ax.plot(avg_rewards)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            plt.ylim([0, self.DURATION])
            plt.xlim([0, self.NUM_EPISODES])
            plt.savefig(self.DATA_PATH)

    def show_simulation(self):
        t = -1

        def policy(time_step):
            nonlocal t
            t += 1
            state = parse(time_step.observation)
            action_a = self.agent.get_action(state, t)
            action_c = self.controller.get_action(state, t)
            action = action_a + action_c
            return action

        viewer.launch(self.env, policy=policy)
