from ddpg_classes.simulator import Simulation
from simulation.dm_control_cur.utility_classes.parameterizer import Parameterizer


class SimulationArm(Simulation):
    # put in environment etc...
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.parameterizer = Parameterizer()

