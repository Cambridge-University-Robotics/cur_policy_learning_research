from ddpg_classes.simulator import Simulation

args = {
    'load_model': False,
    'plot': True,
    'name_model': 'humanoid',
    'task': 'stand',
    'num_episodes': 1000,
    # 'num_episodes': 200,
    'batch_size': 128,
    'duration': 100,
    'gamma': 0.9999,
}
s = Simulation(**args)
s.train()
s.show_simulation()
