from ddpg_classes.simulator import Simulation
from ddpg_classes.simulator_residual import ResidualSimulation

# Simulation Tests
#   Training from scratch
sim = Simulation(
    load_model=False,
    plot=True,
    name_model='cartpole',
    task='balance',
    label=None,
    num_episodes=1,
    batch_size=128,
    duration=500,

)
sim.train()

#   Loading a model
sim = Simulation(
    load_model=True,
    plot=True,
    name_model='cartpole',
    task='balance',
    label=None,
    num_episodes=1,
    batch_size=128,
    duration=500,

)
sim.train()

# Residual Simulation Tests
#   Training from scratch
res = ResidualSimulation(
    # controller options
    controller_load_model=False,
    controller_num_episodes=1,

    # simulation options
    label='residual',
    load_model=False,
    plot=True,
    name_model='cartpole',
    task='balance',
    num_episodes=1,
    batch_size=128,
    duration=100,
)
res.train_controller()
res.train()
#   Loading controller and residual
res = ResidualSimulation(
    # controller options
    controller_load_model=True,
    controller_num_episodes=1,

    # simulation options
    label='residual',
    load_model=True,
    plot=True,
    name_model='cartpole',
    task='balance',
    num_episodes=1,
    batch_size=128,
    duration=100,
)
