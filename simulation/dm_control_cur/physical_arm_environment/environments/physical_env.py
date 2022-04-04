import numpy as np
import collections
import dm_env
from dm_env import specs, TimeStep

from simulation.dm_control_cur.physical_arm_environment.utils.arm import FireflyArm
from simulation.dm_control_cur.physical_arm_environment.utils.object import Object
from simulation.dm_control_cur.utility_classes.abstract_classes import Environment

# TODO: Measure and change ALL the dimension below
# Moving the arm in a plane: fixed Y, yaw, roll
# positive x is right as seen by the robot
# positive y is in front of the robot
# positive z is vertically upward
OBJECT_INITIAL_HEIGHT = 0.46163282  # for the reward function
OBJECT_X_POSITION = 0
OBJECT_Y_POSITION = 100
FIXED_Y_PLANE = 100  # Robotic arm is confined to move in this Y plane
OUTER_BOUNDING_BOX_X_RANGE = (-5000,5000)  # Control the size of Y plane
OUTER_BOUNDING_BOX_Z_RANGE = (0,5000)  # Control the size of Y plane
PROHIBITED_X_RANGE = (-10,10)  # Occupied by object holder
PROHIBITED_Z_RANGE = (10,20)  # Occupied by object holder


class PhysicalEnv(Environment):
    def __init__(self,
                 physics,
                 task,
                 time_limit=float('inf'),
                 control_timestep=None,
                 n_sub_steps=None,
                 flat_observation=False):
        """
        To instantiate this class, use PhysicalEnv.load(...)
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

        # Important note: task and physics are used for virtual env
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
        # else:
        #     self._step_limit = time_limit / (
        #             self._physics.timestep() * self._n_sub_steps)
        self._step_count = 0
        self._reset_next_step = True

        # Load the arm and object
        self.arm = FireflyArm()
        self.object = Object()
        self.arm.connect(port='/dev/ttyUSB0')
        self.arm.calibrate()
        self.object.connect()

        # TODO: Choose a suitable initial x and z positions
        # Move robotic arm to the fixed y plane at the start
        self.arm.commands_print('CARTESIAN', f'1000 {FIXED_Y_PLANE} 3000 MOVETO')

    # IMPORTANT NOTE: load() is used to instantiate PhysicalEnv in simulator.py
    def load(self, name_model, task, task_kwargs):
        return PhysicalEnv(None,None)

    def action_spec(self):
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        # NOTE: BoundedArray of length 2, indicate X and Z relative movement
        return specs.BoundedArray(shape=(2,), dtype=np.int, minimum=-1., maximum=1.)

    def observation_spec(self):
        """
        Returns the observation specification for this environment.
        Infers the spec from the observation, unless the Task implements the
        `observation_spec` method.
        Returns:
            OrderedDict of ArraySpecs describing the shape and data type
            of each corresponding observation.
        """
        obs = collections.OrderedDict()
        obs['grip_pos'] = specs.Array((3,),dtype=np.int)  # x,y,z
        obs['grip_rot'] = specs.Array((2,),dtype=np.int)  # pitch, roll, NO yaw
        obs['object_pos'] = specs.Array((1,),dtype=np.int)  # height of object
        return obs

    def reset(self) -> TimeStep:
        """
        Starts a new episode and returns the first `TimeStep`.
        Returns:
              A `TimeStep` namedtuple containing:
                step_type: A `StepType` of `FIRST`.
                reward: `None`, indicating the reward is undefined.
                discount: `None`, indicating the discount is undefined.
                observation: A NumPy array, or a nested dict, list or tuple of arrays.
                  Scalar values that can be cast to NumPy arrays (e.g. Python floats)
                  are also valid in place of a scalar array. Must conform to the
                  specification returned by `observation_spec()`.
        """
        self._reset_next_step = False
        self._step_count = 0

        self.arm.commands_print('CARTESIAN', f'1000 {FIXED_Y_PLANE} 3000 MOVETO')

        observation = self.get_observation()
        # if self._flat_observation:
        #     observation = flatten_observation(observation)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=observation)

    def step(self, action) -> TimeStep:
        """
        Updates the environment using the action and returns a `TimeStep`.
            Args:
              action: A NumPy array, or a nested dict, list or tuple of arrays
                corresponding to `action_spec()`.
            Returns:
              A `TimeStep` namedtuple containing:
                step_type: A `StepType` value.
                reward: Reward at this timestep, or None if step_type is
                  `StepType.FIRST`. Must conform to the specification returned by
                  `reward_spec()`.
                discount: A discount in the range [0, 1], or None if step_type is
                  `StepType.FIRST`. Must conform to the specification returned by
                  `discount_spec()`.
                observation: A NumPy array, or a nested dict, list or tuple of arrays.
                  Scalar values that can be cast to NumPy arrays (e.g. Python floats)
                  are also valid in place of a scalar array. Must conform to the
                  specification returned by `observation_spec()`.
        """

        if self._reset_next_step:
            return self.reset()
        # no need
        # self._task.before_step(action, self._physics)
        for _ in range(self._n_sub_steps):
            # Correct any invalid action
            action = self.correcting_action(self.get_observation(), action)
            # NOTE: set_state moves RELATIVE to previous position
            # as action[0]=1 represents 0.1mm, it is multiplied by 10 to get 1mm
            self.arm.set_state(action[0]*10, 0, action[1]*10, 0, 0)

        # no need
        # self._task.after_step(self._physics)

        observation = self.get_observation()

        # Reward function similar to the one in passive_hand.py
        # Consider euclidean distance and object height
        dist = np.sum((np.array(observation["grip_pos"]) - np.array(observation["object_pos"])) ** 2) ** (1 / 2)
        height = observation["object_pos"][2] - OBJECT_INITIAL_HEIGHT
        height *= 50
        reward = (-dist) + height

        # no need
        # if self._flat_observation:
        #     observation = flatten_observation(observation)

        self._step_count += 1

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
        # TODO: Note that discount is always 1.0 here
        return dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, observation)

    def get_observation(self):
        """
        A helper function to return observation
        IMPORTANT NOTE If we remove anything here in obs, observation_spec() must be updated
        Return:
            an OrderedDict of NumPy arrays matching the specification returned by observation_spec()
        """
        obs = collections.OrderedDict()

        grip_pos = [self.arm.get_state()["x"], self.arm.get_state()["y"], self.arm.get_state()["z"]]
        grip_rot = [self.arm.get_state()["pitch"], self.arm.get_state()["roll"]]
        # x, y position of object is important to find euclidean distance of arm and object in reward
        object_pos = [OBJECT_X_POSITION, OBJECT_Y_POSITION, self.object.get_state()["height"]]
        obs['grip_pos'] = grip_pos
        obs['grip_rot'] = grip_rot
        obs['object_pos'] = object_pos
        return obs

    def correcting_action(self, observation, action):
        """
        This function will correct action if
            (1) the action will cause the final position to be outside the allowable outer boundary
            (2) the action will cause the final position to fall with prohibited area (object holder)
        Args:
            observation
            action
        Returns:
            action (corrected)
        """
        corrected_action = action.copy()

        new_x = observation["grip_pos"][0] + action[0]
        # Check x: outer boundary
        if new_x < OUTER_BOUNDING_BOX_X_RANGE[0] or new_x > OUTER_BOUNDING_BOX_X_RANGE[1]:
            corrected_action[0] = 0
        # Check x: prohibited area (object holder)
        elif new_x > PROHIBITED_X_RANGE[0] or new_x < PROHIBITED_X_RANGE[1]:
            corrected_action[0] = 0

        new_z = observation["grip_pos"][2] + action[1]
        # Check z: outer boundary (action[1] gives z relative movement, not action[2])
        if new_z < OUTER_BOUNDING_BOX_Z_RANGE[0] or new_z > OUTER_BOUNDING_BOX_Z_RANGE[1]:
            corrected_action[1] = 0
        # Check z: prohibited area (object holder)
        elif new_z > PROHIBITED_Z_RANGE[0] or new_z < PROHIBITED_Z_RANGE[1]:
            corrected_action[1] = 0

        return corrected_action
