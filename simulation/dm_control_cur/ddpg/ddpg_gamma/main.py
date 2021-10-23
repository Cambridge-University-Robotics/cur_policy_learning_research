from ddpg_classes.simulator import Simulation
from simulation.dm_control_cur.utility_classes import data_utilities

for gamma in [0.7, 0.9, 0.95, 0.99, 0.995, 0.999]:
    g2 = str(gamma).replace('.', ',')  # cannot have '.' in file name
    args = {
        'label': f'gamma={g2}',
        'load_model': False,
        'plot': True,
        'name_model': 'cartpole',
        'task': 'balance',
        # 'num_episodes': 5,
        'num_episodes': 200,
        'batch_size': 128,
        'duration': 200,
        'gamma': gamma,
    }
    sim = Simulation(**args)
    sim.train()

data_utilities.plot_aggregate()
