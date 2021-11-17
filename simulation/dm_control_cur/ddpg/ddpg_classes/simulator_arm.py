from simulation.dm_control_cur.utility_classes.simulator import Simulation
from simulation.dm_control_cur.utility_classes.parameterizer import Parameterizer
import numpy as np


class SimulationArm(Simulation, Parameterizer):
    def __init__(self, **kwargs):
        Simulation.__init__(self, **kwargs)
        Parameterizer.__init__(self)

    def modify_obs(self, obs):
        x = np.array([])
        for k, v in obs.items():
            if k == 'simulation_time': continue
            x = np.append(x, v)
        return x
