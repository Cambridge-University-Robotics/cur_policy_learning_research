from simulation.dm_control_cur.virtual_arm_environment import environments
import numpy as np
from dm_control import viewer

#see tasks and domains
get_task = environments._get_tasks(tag=None)
print(get_task)

# load task arm and domain
#env = environments.load(domain_name='passive_hand', task_name='lift_sparse')
# load task humanoid
# env = environments.load(domain_name='humanoid', task_name='stand')
# action_spec = env.action_spec()


# def random_policy(time_step):
#     del time_step
#     # Find random policy based on action_spec
#     return np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)

# # Launch interactive viewer
# viewer.launch(env, policy=random_policy)

# # Load task and domain
env = environments.load(domain_name='passive_hand', task_name='lift_sparse')
print(env)
# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()

while not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                               action_spec.maximum,
                               size=action_spec.shape)
    time_step = env.step(action)
    #print(time_step)
#viewer.launch(env, policy=action)