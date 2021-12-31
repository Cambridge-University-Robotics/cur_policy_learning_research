from simulation.dm_control_cur.utility_classes.abstract_classes import Environment


class PhysicalEnv(Environment):
    """
    Wrapper environment for the robotic arm
    Still working on it
    """

    # To-do
    def load(self, name_model, task, task_kwargs):
        pass

    def action_spec(self):
        pass

    def observation_spec(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

