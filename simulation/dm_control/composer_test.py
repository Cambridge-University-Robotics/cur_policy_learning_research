from simulation_control.environments.passive_hand_composer import *

obj = Object()
robot = PassiveHandRobot()

task = Lift(robot, obj)
env = composer.Environment(task)
