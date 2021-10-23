from pathlib import Path
import shutil
from ddpg_classes.simulator import Simulation
from ddpg_classes.simulator_residual import ResidualSimulation
import json
import logging

logging.basicConfig(filename='unit_tests.log', level=logging.DEBUG)
logging.info("Running unit tests")
logging.info("Removing old directories")
a = [Path('./data'), Path('./models')]
for p in a:
    if p.is_dir():
        logging.info(f"Removed {p}")
        shutil.rmtree(p)

# Simulation Tests
logging.info("Running Simulation tests")
#   Training from scratch
logging.info("Training a Simulation model from scratch")
sim = Simulation(
    load_model=False,
    plot=True,
    name_model='cartpole',
    task='balance',
    label=None,
    num_episodes=1,
    batch_size=128,
    duration=100,
    gamma=0.995,

)
sim.train()

#   Loading a model
logging.info("Loading a Simulation model")
sim = Simulation(
    load_model=True,
    plot=True,
    name_model='cartpole',
    task='balance',
    label=None,
    num_episodes=1,
    batch_size=128,
    duration=100,
)
sim.train()

#   Training from scratch
logging.info("Checking if training performance has been affected")
sim = Simulation(
    load_model=False,
    plot=True,
    name_model='cartpole',
    task='balance',
    label=None,
    num_episodes=100,
    batch_size=128,
    duration=200,
    gamma=0.995,
)
sim.train()
for p in Path('./data').iterdir():
    if p.suffix == '.json':
        f = p.open()
        g = json.load(f)
        h = g['avg_rewards'][-20:]
        avg = sum(h)/len(h)
        if avg < 130:
            logging.warning(f"Average values are low! avg={avg}")



# Residual Simulation Tests
#   Training from scratch
logging.info("Training a Residual Simulation model from scratch")
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
logging.info("Loading Residual Simulation controller and model")
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
