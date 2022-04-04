from simulation.dm_control_cur.utility_classes.simulator import Simulation
from simulation.dm_control_cur.physical_arm_environment.environments.physical_env import PhysicalEnv

sa = Simulation(
    load_model=True,
    plot=True,
    label=None,  # tag that is appended to file name for models and graphs
    num_episodes=1000,  # number of simulation rounds before training session
    batch_size=128,  # number of past simulations to use for training
    duration=200,  # duration of simulation

    env=PhysicalEnv,  # used if we inject an environment
)
# sa.update_xml(obj_amt=0, rbt_amt=0)
sa.train()
sa.show_simulation()