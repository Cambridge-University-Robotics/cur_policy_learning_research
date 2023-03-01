from dm_control import viewer
from dm_env import TimeStep, Environment
from simulation.dm_control_cur.utility_classes import data_wrappers
import numpy as np
from controller import RobotController, SplineTrajectory, LinearTrajectory
from simulation_api import SimulationAPI

# env = virtual_arm_environment.load(domain_name='passive_hand', task_name='lift_sparse', task_kwargs={'time_limit': float('inf')})  # type: Environment
simulation_api = SimulationAPI()
parameters = data_wrappers.EnvironmentParametrization({'object_change_slope': 0.0})

simulation_api.reset(parameters=parameters, task_parameters={'time_limit': float('20')})

controller = RobotController()
trajectory = SplineTrajectory()
trajectory.add_state(pos=[0.0, 0.0, 0.0], vert_rot=0, twist_rot=0)
trajectory.add_state(pos=[1.25, 0.74909766, 0.46112417], vert_rot=0, twist_rot=0)
trajectory.add_state(pos=[1.25, 1, 0.4], vert_rot=0, twist_rot=0)
trajectory.add_state(pos=[1.4, 1., 0.4], vert_rot=0, twist_rot=0)

trajectory.add_state(pos=[1.0, 1.0, 1.0], vert_rot=0, twist_rot=0)
# trajectory.add_state(pos=[1.46177789, 0.84909766, 0.46112417], vert_rot=0, twist_rot=-1.57)
controller.add_trajectory(trajectory, 15)
# liftTrajectory = LinearTrajectory()
# liftTrajectory.add_state(pos=[1.25, 0.74909766, 0.46112417], vert_rot=0, twist_rot=0)
# liftTrajectory.add_state(pos=[1.25, 1, 0.4], vert_rot=0, twist_rot=0)
# liftTrajectory.add_state(pos=[1.4, 1., 0.4], vert_rot=0, twist_rot=0)
# liftTrajectory.add_state(pos=[1.4, 0.74909766, 0.4], vert_rot=0, twist_rot=0)
# liftTrajectory.add_state(pos=[1.4, 0.74909766, 0.4], vert_rot=0, twist_rot=-0.7)
# liftTrajectory.add_state(pos=[0.0, 0.0, 0.0], vert_rot=90, twist_rot=0)
# liftTrajectory.add_state(pos=[1.4, 0.74909766, 1], vert_rot=0, twist_rot=0)
# controller.add_trajectory(liftTrajectory, 12)

def controller_policy(time_step: TimeStep):
    readings = data_wrappers.SensorsReading(time_step.observation)
    print(time_step.observation['object_contact_force'])
    action = controller.get_action(readings)
    return action


debug_action = np.zeros(shape=5)


class DebugController:
    def __init__(self):
        self.debug_action = np.zeros(shape=5)

    def debug_policy(self, time_step: TimeStep):
        print(f'Action taken: {self.debug_action}')
        return self.debug_action

    def set_debug_action(self, new_action):
        self.debug_action = np.arange(new_action)


debug_controller = DebugController()


def debug_policy(time_step: TimeStep):
    print(f'Action taken: {debug_action}')
    return debug_action


viewer.launch(simulation_api.env, policy=debug_controller.debug_policy)


# viewer.launch(simulation_api.env, policy=controller_policy)
