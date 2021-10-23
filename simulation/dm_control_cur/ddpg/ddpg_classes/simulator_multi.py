from abc import ABC, abstractmethod


class AbstractMultiSimulator(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def plot(self):
        pass


class MultiSimulator(AbstractMultiSimulator):
    def __init__(self, simulator_list):
        pass

    def train(self):
        pass

    def plot(self):
        pass
