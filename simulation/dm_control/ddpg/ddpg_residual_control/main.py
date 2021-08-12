import matplotlib.pyplot as plt
import numpy as np
from dm_control import viewer
from tqdm import tqdm

from ddpg_classes.simulator import Simulation, parse


class ResidualSimulation(Simulation):
    def __init__(
            self,
            train_controller=True,
            controller_show=True,
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
            if controller_show:
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
            action_a = self.agent.get_action(state, t)
            action_c = self.controller.get_action(state, t)
            action = action_a + action_c
            return action

        viewer.launch(self.env, policy=policy)


res = ResidualSimulation(
    train_controller=True,
    controller_num_episodes=25,
    controller_show=False,
    label='residual',

    load_model=False,
    plot=True,
    show_simulation=True,
    name_model='cartpole',
    task='balance',
    num_episodes=100,
    batch_size=128,  # number of past simulations to use for training
    duration=200,

)

res.train()
# res.show_simulation()

sim = Simulation(
    load_model=False,
    plot=True,
    show_simulation=True,
    name_model='cartpole',
    task='balance',
    label=None,
    num_episodes=100,
    batch_size=128,  # number of past simulations to use for training
    duration=200,

)
sim.train()

