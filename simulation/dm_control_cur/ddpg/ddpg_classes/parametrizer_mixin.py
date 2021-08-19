from abc import ABC, abstractmethod

from simulation.dm_control_cur.utility_classes.parameterizer import Parameterizer


class AbstractParaMixin(ABC):

    @abstractmethod
    def update_xml(self, obj_amt, robot_amt):
        pass

    @abstractmethod
    def get_params(self):
        pass


class ParaMixin(AbstractParaMixin):
    def __init__(self):
        self.parameterizer = Parameterizer()

    def update_xml(self, obj_amt, robot_amt):
        self.parameterizer._randomize_object(obj_amt)
        self.parameterizer._randomize_robot(robot_amt)
        self.parameterizer.update_xml()

    def get_params(self):
        return self.parameterizer.get_parameters()
