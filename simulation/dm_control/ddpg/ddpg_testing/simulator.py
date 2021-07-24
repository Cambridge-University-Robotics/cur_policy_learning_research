from simulation.dm_control.ddpg.ddpg_classes.ddpg import DDPGagent, OUNoise

"""
This script will run training for different models/configurations on different
control suite tasks so that we can verify that some models are better than others
"""
TASKS_TO_RUN = [
    ('cartpole', 'balance'),
    ('cartpole', 'swingup'),
]
