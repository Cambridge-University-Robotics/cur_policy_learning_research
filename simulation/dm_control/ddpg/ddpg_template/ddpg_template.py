"""
Template Document
If you want to try something, copy this document
"""
import numpy as np
import matplotlib.pyplot as plt
from dm_control import suite, viewer
from tqdm import tqdm
from simulation.dm_control.ddpg.ddpg_classes.ddpg import DDPGagent
from simulation.dm_control.ddpg.ddpg_classes.utils import MemorySeq
from datetime import datetime

LOAD_MODEL = False
TRAIN = True
SHOW_SIMULATION = True

random_state = np.random.RandomState(42)
DATE_TIME = datetime.now().strftime("%d:%m:%Y-%H:%M:%S")
NAME_MODEL = 'cartpole'
TASK = 'balance'
MODEL_PATH = f'models/{NAME_MODEL}_{TASK}'
DATA_PATH = f'data/{NAME_MODEL}_{TASK}_{DATE_TIME}'
NUM_EPISODES = 50
BATCH_SIZE = 128
DURATION = 50
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
GAMMA = 0.99
TAU = 1e-2
env = suite.load(NAME_MODEL, TASK, task_kwargs={'random': random_state})
action_spec = env.action_spec()
obs_spec = env.observation_spec()
dim_action = action_spec.shape[0]
dim_obs = sum(tuple(map(lambda x: int(np.prod(x.shape)), obs_spec.values())))

agent = DDPGagent(
    num_states=dim_obs,
    num_actions=dim_action,
    action_low=action_spec.minimum,
    action_high=action_spec.maximum,
    actor_learning_rate=ACTOR_LEARNING_RATE,
    critic_learning_rate=CRITIC_LEARNING_RATE,
    gamma=GAMMA,
    tau=TAU,
    memory=MemorySeq
)
if LOAD_MODEL: agent.load(MODEL_PATH)


def parse(obs):
    """
    We lose information about the variables when we combine, for instance,
    velocity and position into 1 long array of observations. Could treating
    them separately lead to a better NN architecture? Velocity, position
    and numerical integration for instance take only ...
    """
    x = np.array([])
    for _, v in obs.items():
        x = np.append(x, v)
    return x


if TRAIN:
    rewards = []
    avg_rewards = []

    tqdm_range = tqdm(range(NUM_EPISODES))
    for episode in tqdm_range:
        time_step = env.reset()
        state = parse(time_step.observation)
        episode_reward = 0

        for t in range(DURATION):
            action = agent.get_action(state, t=t)
            time_step_2 = env.step(action)
            state_2 = parse(time_step_2.observation)
            agent.push(state, action, time_step_2.reward, state_2, -1)
            state = state_2
            agent.update(BATCH_SIZE)
            episode_reward += time_step_2.reward

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
        desc = f"episode: {episode}, " \
               f"reward: {np.round(episode_reward, decimals=2)}, " \
               f"average_reward: {np.mean(rewards[-10:])}"
        tqdm_range.set_postfix_str(s=desc)

    agent.save(MODEL_PATH)

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(DATA_PATH)
    plt.show()

if SHOW_SIMULATION:
    t = -1


    def policy(time_step):
        global t
        t += 1
        state = parse(time_step.observation)
        action = agent.get_action(state, t)
        return action


    viewer.launch(env, policy=policy)
