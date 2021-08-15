from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from ..utils import mjdata_utils
import os

ASSETS_PATH = 'passive_hand'

class TableArena(composer.Arena):
    def _build(self):
        with open(os.path.join(ASSETS_PATH, 'scene.xml')) as f:
            model = mjcf.from_file(f)
        return model

class PassiveHand(composer.Entity):
    def _build(self):
        with open(os.path.join(ASSETS_PATH, 'robot.xml')) as f:
            model = mjcf.from_file(f)
        return model

    def _build_observables(self):
        return PassiveHandObservables(self)


class Object(composer.Entity):
    def _build(self):
        with open(os.path.join(ASSETS_PATH, 'object.xml')) as f:
            model = mjcf.from_file(f)
        return model

    def _build_observables(self):
        return ObjectObservables(self)


class PassiveHandObservables(composer.Observables):
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
    def object_velp(self):
        return observable.Generic(lambda x: mjdata_utils.get_site_vel('object0', False, x))[3:]

    @composer.observable
    def object_velr(self):
        return observable.Generic(lambda x: mjdata_utils.get_site_vel('object0', False, x))[:3]

    @composer.observable
    def object_pos(self):
        obj = self._entity.mjcf_model.find('site', 'object0')
        return observable.MJCFFeature('site_xpos', obj)

    @composer.observable
    def object_pos(self):
        obj = self._entity.mjcf_model.find('site', 'object0')
        return observable.MJCFFeature('site_xpos', obj)

    @composer.observable
    def object_rot(self):
        return observable.Generic(lambda x: mjdata_utils.get_site_rot_mat('object0', x))


class Lift(composer.Task):
    def __init__(self, robot, obj):
        self._robot = robot
        self._object = obj
        self._arena = TableArena()
