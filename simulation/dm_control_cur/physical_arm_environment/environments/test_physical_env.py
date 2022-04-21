from simulation.dm_control_cur.physical_arm_environment.environments.physical_env import PhysicalEnv
from time import sleep
import collections


OUTER_BOUNDING_BOX_X_RANGE = (-400,400)  # Control the size of Y plane
OUTER_BOUNDING_BOX_Z_RANGE = (300,800)  # Control the size of Y plane
PROHIBITED_X_RANGE = (-80,80)  # Occupied by object holder
PROHIBITED_Z_RANGE = (300,450)  # Occupied by object holder
MAX_STEP_RANGE = 10


def correcting_action(observation, action):
    """
    This function will correct action if
        (1) the action will cause the final position to be outside the allowable outer boundary
        (2) the action will cause the final position to fall within prohibited area (object holder)
    Args:
        observation
        action
    Returns:
        action (corrected)
    """
    corrected_action = action

    new_x = float(observation["grip_pos"][0]) + float(action[0] * MAX_STEP_RANGE)
    # Check x: outer boundary
    if new_x < OUTER_BOUNDING_BOX_X_RANGE[0] or new_x > OUTER_BOUNDING_BOX_X_RANGE[1]:
        corrected_action[0] = 0
    # Check x: prohibited area (object holder)
    elif new_x > PROHIBITED_X_RANGE[0] and new_x < PROHIBITED_X_RANGE[1]:
        corrected_action[0] = 0

    new_z = float(observation["grip_pos"][2]) + float(action[1] * MAX_STEP_RANGE)
    # Check z: outer boundary (action[1] gives z relative movement, not action[2])
    if new_z < OUTER_BOUNDING_BOX_Z_RANGE[0] or new_z > OUTER_BOUNDING_BOX_Z_RANGE[1]:
        corrected_action[1] = 0
    # Check z: prohibited area (object holder)
    elif new_z > PROHIBITED_Z_RANGE[0] and new_z < PROHIBITED_Z_RANGE[1]:
        corrected_action[1] = 0

    return corrected_action


def test_correcting_action():
    obs = collections.OrderedDict()

    # Test 1
    grip_pos = [100, 100, 500]
    grip_rot = [0, 0]
    object_pos = [0,0,0]
    obs['grip_pos'] = grip_pos
    obs['grip_rot'] = grip_rot
    obs['object_pos'] = object_pos
    action = [1,0]
    assert correcting_action(obs, action) == [1,0]

    # Test 2
    grip_pos = [81, 100, 500]
    grip_rot = [0, 0]
    object_pos = [0, 0, 0]
    obs['grip_pos'] = grip_pos
    obs['grip_rot'] = grip_rot
    obs['object_pos'] = object_pos
    action = [-1,0]
    assert correcting_action(obs, action) == [0,0]

    # Test 3
    grip_pos = [399, 100, 500]
    grip_rot = [0, 0]
    object_pos = [0, 0, 0]
    obs['grip_pos'] = grip_pos
    obs['grip_rot'] = grip_rot
    obs['object_pos'] = object_pos
    action = [-1, 0]
    assert correcting_action(obs, action) == [-1, 0]

    # Test 4
    grip_pos = [399, 100, 500]
    grip_rot = [0, 0]
    object_pos = [0, 0, 0]
    obs['grip_pos'] = grip_pos
    obs['grip_rot'] = grip_rot
    obs['object_pos'] = object_pos
    action = [1, 0]
    assert correcting_action(obs, action) == [0, 0]