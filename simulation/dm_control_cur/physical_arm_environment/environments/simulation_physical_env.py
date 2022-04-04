from simulation.dm_control_cur.utility_classes.simulator import Simulation
from simulation.dm_control_cur.utility_classes.parameterizer import Parameterizer
import numpy as np
from simulation.dm_control_cur.physical_arm_environment.environments.physical_arm_env import PhysicalEnv


class SimulationPhysicalArm(Simulation, Parameterizer):
    def __init__(self, **kwargs):
        Simulation.__init__(self, **kwargs)
        Parameterizer.__init__(self)

    def modify_obs(self, obs):
        x = np.array([])
        for k, v in obs.items():
            if k == 'simulation_time': continue
            x = np.append(x, v)
        return x
    def physical_movement(self):
        t = -1
        time_step = self.env.reset()
        while(True):
            t += 1
            
            state = self.modify_obs(time_step.observation)
            action = self.agent.get_action(state, t)
            
            time_step = self.env.step(action)
            #receive feedback from the arm stating that action is done
            #set new state.
env = PhysicalEnv()
sa = SimulationPhysicalArm(
    load_model=False,
    plot=False,
    name_model='passive_hand',
    task='lift_sparse',
    label=None,  # tag that is appended to file name for models and graphs
    num_episodes=1000,  # number of simulation rounds before training session
    batch_size=128,  # number of past simulations to use for training                       
    duration=200,  # duration of simulation

    env=env,  # used if we inject an environment
    dim_obs=33,
)
# PUT PHYSICAL_ENV() IN A INIT FILE LIKE WHAT VIRTUAL ARM DID, SO THAT WHEN THE SIMULATOR CALL ENV.LOAD(), IT LOADS THE PHYSICAL_ENV() INSTANCE
# sa.train()
sa.physical_movement()