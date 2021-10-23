from ddpg_classes.simulator import Simulation
from ddpg_classes.simulator_residual import ResidualSimulation
import multiprocessing as mp
SIM_CLASS = 'sim_class'
ARGS = 'args'
args_list = [
    {
        SIM_CLASS: ResidualSimulation,
        ARGS: {
            # controller options
            'controller_load_model': False,
            'controller_num_episodes': 2,
            # simulation options
            'label': 'residual',
            'load_model': False,
            'plot': True,
            'name_model': 'cartpole',
            'task': 'balance',
            'num_episodes': 2,
            'batch_size': 128,
            'duration': 500,
        }
    },
    {
        SIM_CLASS: ResidualSimulation,
        ARGS: {
            # controller options
            'controller_load_model': False,
            'controller_num_episodes': 2,
            # simulation options
            'label': 'residual',
            'load_model': False,
            'plot': True,
            'name_model': 'cartpole',
            'task': 'balance',
            'num_episodes': 2,
            'batch_size': 128,
            'duration': 500,
        }
    }
]


# Note: double check load, train, show_simulation values before using!!!
def f(v):
    sim_class = (v[SIM_CLASS])(**(v[ARGS]))
    sim_class.train()

with mp.Pool() as p:
    p.map(func=f, iterable=args_list)
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
sim.show_simulation()
# sim.train()
