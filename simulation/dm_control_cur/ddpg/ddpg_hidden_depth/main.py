from ddpg_classes.simulator import Simulation
from simulation.dm_control_cur.utility_classes import data_utilities

for name_model, task in [
    ('cartpole', 'balance'),
    ('swimmer', 'swimmer6'),
    ('hopper', 'hop'),
    ('cheetah', 'run'),
    ('walker', 'run'),
]:
    for depth in [1, 2, 3, 4, 5]:
        args = {
            'label': f'depth={depth}',
            'load_model': False,
            'plot': True,
            'name_model': name_model,
            'task': task,
            # 'num_episodes': 3,
            'num_episodes': 200,
            'batch_size': 128,
            # 'duration': 50,
            'duration': 200,
            'hidden_depth': depth,
            'hidden_size': 64,
        }
        sim = Simulation(**args)
        sim.train()

    name_model_task = f'{name_model}_{task}'
    data_utilities.plot_aggregate(
        model_name=name_model_task,
        save_name=f'aggregated_{name_model_task}'
    )
