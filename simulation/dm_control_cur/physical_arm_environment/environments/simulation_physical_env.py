from simulation.dm_control_cur.utility_classes.simulator import Simulation
from simulation.dm_control_cur.utility_classes.parameterizer import Parameterizer
import numpy as np
from simulation.dm_control_cur.physical_arm_environment.environments.physical_env import PhysicalEnv


class SimulationPhysicalArm(Simulation, Parameterizer):
    def __init__(self, **kwargs):
        Simulation.__init__(self, **kwargs)
        Parameterizer.__init__(self)

    def show_simulation(self):
        t = -1
        time_step = self.env.reset()
        while(True):
            t += 1
            
            state = self.modify_obs(time_step.observation)
            action = self.agent.get_action(state, t)
            
            time_step = self.env.step(action)
            #receive feedback from the arm stating that action is done
            #set new state.


sa = SimulationPhysicalArm(
    load_model=True,
    plot=False,
    label=None,  # tag that is appended to file name for models and graphs
    num_episodes=1000,  # number of simulation rounds before training session
    batch_size=128,  # number of past simulations to use for training                       
    duration=200,  # duration of simulation

    env=PhysicalEnv,  # used if we inject an environment
)
# PUT PHYSICAL_ENV() IN A INIT FILE LIKE WHAT VIRTUAL ARM DID, SO THAT WHEN THE SIMULATOR CALL ENV.LOAD(), IT LOADS THE PHYSICAL_ENV() INSTANCE
# sa.train()
sa.show_simulation()