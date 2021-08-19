# TODO: port reward function from ../mujoco_gym/hand_optimization.py
# from simulation_control.dm_control_cur.simulation_control import SensorsReading
from simulation.dm_control_cur.utility_classes.data_wrappers import SensorsReading


def placeholder_reward_func(last_reward: float, step: int, last_step: bool, readings: SensorsReading) -> float:
    raise NotImplementedError()