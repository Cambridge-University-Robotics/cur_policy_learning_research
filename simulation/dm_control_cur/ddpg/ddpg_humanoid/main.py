from simulation.dm_control_cur.utility_classes.simulator import Simulation

args = {
    'load_model': True,
    'plot': True,
    'name_model': 'humanoid',
    'task': 'stand',
    # 'num_episodes': 10000,
    'num_episodes': 200,
    'batch_size': 128,
    'duration': 100,
    'gamma': 0.9999,
}
s = Simulation(**args)
#s.train()
s.show_simulation()
