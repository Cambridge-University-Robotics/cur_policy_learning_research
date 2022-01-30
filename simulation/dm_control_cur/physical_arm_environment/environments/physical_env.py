from simulation.dm_control_cur.utility_classes.abstract_classes import Environment
from simulation.dm_control_cur.physical_arm_environment.utils.arm import FireflyArm

from dm_env import specs


class PhysicalEnv(Environment):
    """
    Wrapper environment for the robotic arm
    Still working on it
    """

    def load(self, name_model, task, task_kwargs):
        fa = FireflyArm()
        fa.connect(port='/dev/ttyUSB0')
        fa.calibrate()  # arm needs to be calibrated each startup
        # fa.set_state()
        # fa.disconnect()
        # To be completed

    def action_spec(self):
        """
        Returns a `BoundedArray` spec from dm_env
        """
        return specs.BoundedArray(shape=,
                                  dtype=,
                                  minimum=,
                                  maximum=)

    def observation_spec(self):
        """
        Returns:
          An `OrderedDict` mapping observation name to `specs.Array` containing
          observation shape and dtype.
        """
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

