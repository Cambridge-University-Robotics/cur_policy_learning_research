from ddpg_classes.simulator import Simulation
from ddpg_classes.simulator_residual import ResidualSimulation

# Note: double check load, train, show_simulation values before using!!!
# res = ResidualSimulation(
#     # controller options
#     # controller_load_model=True,
#     controller_num_episodes=5,
#
#     # simulation options
#     label='residual',
#     load_model=True,
#     plot=True,
#     name_model='cartpole',
#     task='balance',
#     num_episodes=10,
#     batch_size=128,
#     duration=100,
# )
# res.train_controller()
# res.train()
# res.show_simulation()

sim = Simulation(
    load_model=False,
    plot=True,
    name_model='cartpole',
    task='balance',
    label=None,
    num_episodes=100,
    batch_size=128,
    duration=500,

)
sim.show_simulation()
# sim.train()
