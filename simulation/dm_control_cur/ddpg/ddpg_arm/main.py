from ddpg_classes.simulator_arm import SimulationArm
import simulation.dm_control_cur.simulation_control.environments as env

sa = SimulationArm(
    load_model=True,
    plot=True,
    name_model='passive_hand',
    task='lift_sparse',
    label=None,  # tag that is appended to file name for models and graphs
    num_episodes=1000,  # number of simulation rounds before training session
    batch_size=128,  # number of past simulations to use for training
    duration=200,  # duration of simulation

    env=env,  # used if we inject an environment
    dim_obs=33,
)
sa.update_xml(obj_amt=0, rbt_amt=0)
sa.train()
sa.show_simulation()
