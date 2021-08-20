import matplotlib.pyplot as plt
import numpy as np
from dm_control import viewer
from tqdm import tqdm

from ddpg_classes.simulator import Simulation


class ResidualSimulation(Simulation):
    def __init__(
            self,
            controller_load_model=True,
            controller_num_episodes=50,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.controller = Simulation(
            load_model=controller_load_model,
            label='controller',
            name_model=self.NAME_MODEL,
            task=self.TASK,
            num_episodes=controller_num_episodes,
            batch_size=self.BATCH_SIZE,
            duration=self.DURATION,
        )

    def train_controller(self):
        self.controller.train()

    def show_controller_simulation(self):
        self.controller.show_simulation()

    def modify_action(self, action, state, t):
        return self.controller.get_action(state, t)
