from ddpg_classes.simulator import Simulation
sim = Simulation(
    load_model=True,
    plot=True,
    name_model='cartpole',
    task='balance',
    label=None,
    num_episodes=100,
    batch_size=128,  # number of past simulations to use for training
    duration=500,
)
