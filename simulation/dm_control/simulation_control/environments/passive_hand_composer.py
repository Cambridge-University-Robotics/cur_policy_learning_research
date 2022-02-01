from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from ..utils import mjdata_utils, mocap_utils, rotations, pymjcf_utils
from .. import constants
from dm_control.composer import variation
from dm_env import specs
import os
import numpy as np

class TableArena(composer.Arena):
    def _build(self):
        with open(os.path.join(constants.ASSETS_PATH, 'passive_hand', 'scene.xml')) as f:
            model = mjcf.from_file(f)
        self._model = model

    @property
    def mjcf_model(self):
        return self._model

class PassiveHandRobot(composer.Entity):
    def _build(self):
        model = mjcf.from_path(os.path.join(constants.ASSETS_PATH, 'passive_hand', 'robot.xml'))
        self._model = model

    @property
    def mjcf_model(self):
        return self._model

    def _build_observables(self):
        return PassiveHandRobotObservables(self)


class Object(composer.Entity):
    def _build(self):
        self._model = mjcf.RootElement()
        self._model.asset.add('material', name='block_mat', specular=0, shininess=0.5, reflectance=0,
                                  rgba=[0.2, 0.2, 0.2, 1])
        self._model.worldbody.add('site', name='object0', type='sphere', size=[0.02, 0.02, 0.10],
                                            rgba=[1, 0, 0, 1], pos=[0, 0, 0])
        self._model.worldbody.add('geom', name='object0', type='cylinder', condim=3, material='block_mat',
                                            mass=0.5, size=[0.05, 0.10], pos=[0, 0, 0])

    @property
    def mjcf_model(self):
        return self._model

    def _build_observables(self):
        return ObjectObservables(self)


class PassiveHandRobotObservables(composer.Observables):
    @composer.observable
    def grip_position(self):
        grip = self._entity.mjcf_model.find('site', 'robot0:grip')
        return observable.MJCFFeature('site_xpos', grip)

    @composer.observable
    def grip_vel(self):
        return observable.Generic(lambda x: mjdata_utils.get_site_vel('robot0:grip', False, x))

    @composer.observable
    def grip_rot(self):
        return observable.Generic(lambda x: mjdata_utils.get_site_rot_mat('robot0:grip', x))


class ObjectObservables(composer.Observables):
    @composer.observable
    def velp(self):
        return observable.Generic(lambda x: mjdata_utils.get_site_vel('object0', False, x)[3:])

    @composer.observable
    def velr(self):
        return observable.Generic(lambda x: mjdata_utils.get_site_vel('object0', False, x)[:3])

    @composer.observable
    def position(self):
        obj = self._entity.mjcf_model.find('site', 'object0')
        return observable.MJCFFeature('site_xpos', obj)

    @composer.observable
    def position(self):
        obj = self._entity.mjcf_model.find('site', 'object0')
        return observable.MJCFFeature('site_xpos', obj)

    @composer.observable
    def rotation(self):
        return observable.Generic(lambda x: mjdata_utils.get_site_rot_mat('object0', x))


class Lift(composer.Task):
    def __init__(self, robot: PassiveHandRobot, obj: Object, sparse=False):
        self._robot = robot
        self._object = obj
        self._arena = TableArena()
        self._object_attachement_frame = self._arena.add_free_entity(self._object)
        self._robot_attachment_frame = self._arena.attach(self._robot)

        # add mocap for robot control
        mocap = self._arena.mjcf_model.worldbody.add('body', name='robot0:mocap', pos=[0, 0, 0], mocap=True)
        self._arena.mjcf_model.equality.add('weld', body1=mocap,
                                            body2=self._robot.mjcf_model.find('body', 'robot0:gripper_link'),
                                            solimp='0.1 0.5 0.1 0.5 6', solref='0.02 1')

        # add contact constraints for robot with object
        object_geoms = self._object.mjcf_model.find_all('geom')
        robot_geoms = self._robot.mjcf_model.find_all('geom')

        for object_geom in object_geoms:
            for robot_geom in robot_geoms:
                self._arena.mjcf_model.contact.add('pair', geom1=object_geom, geom2=robot_geom,
                                                   solimp=[0.9, 0.95, 0.001, 0.5, 2], solref=[0.02, 1])

        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        self.object_initial_height = 0.45

        self._object_initial_pos = (1.46177789, 0.74909766, self.object_initial_height)

        self.gripper_extra_height = 0.2
        self.sparse_reward = sparse
        self.initial_qpos = {
            self._robot_attachment_frame.full_identifier + 'robot0:slide0': 0.405,
            self._robot_attachment_frame.full_identifier + 'robot0:slide1': 0.48,
            self._robot_attachment_frame.full_identifier + 'robot0:slide2': 0.0,

            # I FOUND THE BUG :DDD 5H OF SEARCHING OMGGGG
            # PLS NEVER HARDCODE AGAIN :'(((((((((((
            # 'object0:joint': 0.0,
        }

    @property
    def root_entity(self):
        return self._arena

    def action_spec(self, physics):
        '''Since the robot is controlled by a mocap, the action spec matches the dimensions of the mocap'''
        return specs.BoundedArray(shape=(5,), dtype=np.float, minimum=-1., maximum=1.)

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)

        self._object.set_pose(physics, position=(1.46177789, 0.74909766, self.object_initial_height), quaternion=[0, 0, 0, 0])

        for name, value in self.initial_qpos.items():
            physics.named.data.qpos[name] = value
        mocap_utils.reset_mocap_welds(physics)

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.10, -0.531 + self.gripper_extra_height]) \
                         + physics.named.data.site_xpos[self._robot_attachment_frame.full_identifier + 'robot0:grip']
        # gripper_target = np.array([20., 0., 0.])

        # commenting out the line below to stop the console spam
        # print('Ideal Start Position: ', gripper_target)

        gripper_rotation = np.array([1., 0., 0., 0.])
        physics.named.data.mocap_pos['robot0:mocap'] = gripper_target
        physics.named.data.mocap_quat['robot0:mocap'] = gripper_rotation

        # change bitmasks for geoms to stop gripper from colliding with object in initialization
        n_geoms = physics.named.model.body_geomnum[self._object_attachement_frame.full_identifier]
        start_geom = physics.named.model.body_geomadr[self._object_attachement_frame.full_identifier]
        init_contype = physics.model.geom_contype[start_geom]
        init_conaffinity = physics.model.geom_conaffinity[start_geom]
        target_contype = int(1 << 1)
        # print('Robot Contype and Conaffinity: {} {}'.format(
        #     physics.named.model.geom_contype[self._robot_attachment_frame.full_identifier + 'robot0:gripper_link'],
        #     physics.named.model.geom_conaffinity[self._robot_attachment_frame.full_identifier + 'robot0:gripper_link']))
        # print('Target Contype:', target_contype)

        for i in range(start_geom, start_geom + n_geoms):
            physics.model.geom_contype[i] = target_contype
            physics.model.geom_conaffinity[i] = target_contype
            # print(physics.model.id2name(i, 'geom'))

        for _ in range(50):
            physics.step()

        for i in range(start_geom, start_geom + n_geoms):
            physics.model.geom_contype[i] = init_contype
            physics.model.geom_conaffinity[i] = init_conaffinity

    def before_step(self, physics, action, random_state):
        """Sets the control signal for the actuators to values in `action`."""
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl = action[:3]

        # arm origin at [[0.725, 0.74910034]]
        arm_origin = np.asarray([0.725, 0.74910034])
        vertical_angle = action[3]
        horizontal_angle = -np.arctan2(*(physics.named.data.mocap_pos['robot0:mocap'][:2] - arm_origin)) + np.pi / 2
        twist_angle = action[4]

        rot_ctrl = rotations.quat_mul(rotations.quat_mul(rotations.euler2quat([0., 0., horizontal_angle]),
                                                         rotations.euler2quat([0., vertical_angle, 0.])),
                                      rotations.euler2quat([twist_angle, 0., 0.]))

        pos_ctrl *= 0.05  # limit maximum change in position
        action = np.concatenate([pos_ctrl, rot_ctrl])
        mocap_utils.mocap_set_action(physics, action)

    def get_reward(self, physics):
        grip_pos = physics.named.data.site_xpos[self._robot_attachment_frame.full_identifier + 'robot0:grip']
        object_pos = physics.named.data.site_xpos[self._object_attachement_frame.full_identifier + 'object0']
        dist = np.sum((grip_pos - object_pos) ** 2) ** (1 / 2)  # euclidean distance
        height = object_pos[2] - self.object_initial_height
        height = height * 50
        reward = height + 0 if self.sparse_reward else (-dist)
        return reward

