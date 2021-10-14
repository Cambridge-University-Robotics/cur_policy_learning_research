import numpy as np
from dm_control.mujoco.wrapper import mjbindings

mjlib = mjbindings.mjlib
mjtObj = mjbindings.enums.mjtObj


def get_site_vel(site_name, is_local, physics):
    # 6DOF Vector with first 3 as velr and second 3 as velp
    vels = np.zeros(6)
    mjlib.mj_objectVelocity(physics.model.ptr, physics.data.ptr, mjtObj.mjOBJ_SITE,
                            physics.model.name2id(site_name, 'site'), vels, int(is_local))
    return vels


def get_site_rot_mat(site_name, physics):
    return physics.named.data.site_xmat[site_name].reshape((3, 3))