from ddpg_classes.simulator import Simulation
from simulation.dm_control_cur.utility_classes import data_utilities

for name_model, task in [
    ('cheetah', 'run'),
    ('walker', 'run'),
]:
    for lr in [1, 2, 3, 4, 5]:
        args = {
            'label': f'lr={lr}',
            'load_model': False,
            'plot': True,
            'name_model': name_model,
            'task': task,
            # 'num_episodes': 3,
            'num_episodes': 500,
            'batch_size': 128,
            # 'duration': 50,
            'duration': 200,
            'hidden_size': 64,
            'hidden_depth': 4,
            'actor_learning_rate': lr * 1e-4,
            'critic_learning_rate': lr * 1e-3,
        }
        sim = Simulation(**args)
        sim.train()

    name_model_task = f'{name_model}_{task}'
    data_utilities.plot_aggregate(
        model_name=name_model_task,
        save_name=f'aggregated_{name_model_task}'
    )
