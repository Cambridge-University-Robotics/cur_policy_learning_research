from dm_control.rl import control
from dm_control.utils import containers
import os
import numpy as np
import collections
from simulation.dm_control_cur.virtual_arm_environment.utils import mocap_utils
from simulation.dm_control_cur.virtual_arm_environment.environments import base
from simulation.dm_control_cur.virtual_arm_environment.utils import rotations
from dm_env import specs
from simulation.dm_control_cur.physical_arm_environment.utils.arm import FireflyArm
from simulation.dm_control_cur.physical_arm_environment.utils.object import Object
from simulation.dm_control_cur.utility_classes.abstract_classes import Environment
import dm_env

_DEFAULT_TIME_LIMIT = 15
MODEL_XML_PATH = os.path.join('physical_hand', 'lift.xml')
_N_SUBSTEPS = 20
OBJECT_INITIAL_HEIGHT = 0.46163282

SUITE = containers.TaggedTasks()
# get action from state by agent then use action to find new state in the step() method.
# how to load agent to environment?
# do we need to add it to the suite/ do we need to use it with a simulation class?
# how to get action and ouput state, which method?
class PhysicalEnv(Environment):
    def __init__(self,
               physics,
               task,
               time_limit=float('inf'),
               control_timestep=None,
               n_sub_steps=None,
               flat_observation=False):
    """Initializes a new `Environment`.
    Args:
      physics: Instance of `Physics`.
      task: Instance of `Task`.
      time_limit: Optional `int`, maximum time for each episode in seconds. By
        default this is set to infinite.
      control_timestep: Optional control time-step, in seconds.
      n_sub_steps: Optional number of physical time-steps in one control
        time-step, aka "action repeats". Can only be supplied if
        `control_timestep` is not specified.
      flat_observation: If True, observations will be flattened and concatenated
        into a single numpy array.
    Raises:
      ValueError: If both `n_sub_steps` and `control_timestep` are supplied.
    """
        self._task = task
        self._physics = physics
        self._flat_observation = flat_observation

        if n_sub_steps is not None and control_timestep is not None:
            raise ValueError('Both n_sub_steps and control_timestep were supplied.')
        elif n_sub_steps is not None:
            self._n_sub_steps = n_sub_steps
        # elif control_timestep is not None:
        #     self._n_sub_steps = compute_n_steps(control_timestep,
        #                                   self._physics.timestep())
        else:
            self._n_sub_steps = 1

        if time_limit == float('inf'):
            self._step_limit = float('inf')
        else:
            self._step_limit = time_limit / (
                self._physics.timestep() * self._n_sub_steps)
        self._step_count = 0
        self._reset_next_step = True
        self.arm = FireflyArm()
        self.object = Object()
    #connect to the arm and object
    def load(self, name_model, task, task_kwargs):
        self.arm.connect(port='/dev/ttyUSB0')
        self.arm.calibrate()
        self.object.connect()
        
    
    def action_spec(self):
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        return specs.BoundedArray(shape=(5,), dtype=np.float, minimum=-1., maximum=1.)

    def observation_spec(self):
        return super().observation_spec()

    def reset(self):
        pass

    def step(self, action):
        """Updates the environment using the action and returns a `TimeStep`."""
 
        if self._reset_next_step:
            return self.reset()
        # no need
        # self._task.before_step(action, self._physics)
        for _ in range(self._n_sub_steps):
            # replace by physical step() method
            # self._physics.step()

            self.arm.set_state(*action)
            
        # no need
        # self._task.after_step(self._physics)

        # no need
        # reward = self._task.get_reward(self._physics)
        # change to physical get observation to get all 11 variables or just 6
        # observation = self._task.get_observation(self._physics)
        obs = collections.OrderedDict()
        grip_pos = [self.arm.get_state()["x"], self.arm.get_state()["y"], self.arm.get_state()["z"]]
        grip_velp = [0] * 3
        grip_velr = [0] * 3
        grip_rot = [self.arm.get_state()["pitch"], self.arm.get_state()["roll"], 0]
        object_pos = [0, 0] + self.object.get_state()["height"]
        object_rel_pos = [0] *3
        object_velp = [0] * 3
        object_velr = [0] * 3
        object_rel_velp = [0]  * 3 
        contact_force = [0] * 6 
        obs['grip_pos'] = grip_pos
        obs['grip_velp'] = grip_velp
        obs['grip_velr'] = grip_velr
        obs['grip_rot'] = grip_rot
        obs['object_pos'] = object_pos
        obs['object_rel_pos'] = object_rel_pos
        obs['object_velp'] = object_velp
        obs['object_velr'] = object_velr
        obs['object_rel_velp'] = object_rel_velp
        obs['simulation_time'] = 0
        obs['object_contact_force'] = contact_force
        observation = obs
        reward = self.object.get_state()["height"]
        # no need
        # if self._flat_observation:
        #     observation = flatten_observation(observation)

        # self._step_count += 1
        # no need discount atm
        # if self._step_count >= self._step_limit:
        #     discount = 1.0
        # else:
        #     discount = self._task.get_termination(self._physics)

        # episode_over = discount is not None

        # if episode_over:
        #     self._reset_next_step = True
        #     return dm_env.TimeStep(
        #         dm_env.StepType.LAST, reward, discount, observation)
        # else:
        return dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, observation)  
#make object
#TO DO: safety method