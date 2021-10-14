from simulation_control.environments.passive_hand_composer import *
import numpy as np
from dm_control import viewer

obj = Object()
robot = PassiveHandRobot()

task = Lift(robot, obj)
env = composer.Environment(task)
action_spec = env.action_spec()

def controller_policy(time_step):
    return np.random.random(size=action_spec.shape)


viewer.launch(env, policy=controller_policy)