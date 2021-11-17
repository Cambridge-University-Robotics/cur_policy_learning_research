from simulation.dm_control_cur.utility_classes.simulator import Simulation
from simulation.dm_control_cur.utility_classes import data_utilities

for name_model, task in [
    ('cheetah', 'run'),
    ('walker', 'run'),
]:
    for size in [32, 64, 128, 256]:
        args = {
            'label': f'size={size}',
            'load_model': False,
            'plot': True,
            'name_model': name_model,
            'task': task,
            # 'num_episodes': 3,
            'num_episodes': 500,
            'batch_size': 128,
            # 'duration': 50,
            'duration': 200,
            'hidden_size': size,
        }
        sim = Simulation(**args)
        sim.train()

    name_model_task = f'{name_model}_{task}'
    data_utilities.plot_aggregate(
        model_name=name_model_task,
        save_name=f'aggregated_{name_model_task}'
    )
