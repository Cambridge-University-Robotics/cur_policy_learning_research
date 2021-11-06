from abc import ABC, abstractmethod


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

    @abstractmethod
    def modify_obs(self, obs):
        pass

    @abstractmethod
    def modify_action(self, action, state, t):
        pass
