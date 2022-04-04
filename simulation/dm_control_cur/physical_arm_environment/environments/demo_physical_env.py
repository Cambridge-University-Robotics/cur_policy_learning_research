from simulation.dm_control_cur.physical_arm_environment.environments.physical_env import PhysicalEnv
from time import sleep

if __name__ == "__main__":
    pe = PhysicalEnv(None, None)
    print("Initialised, moving to the left")
    # print(pe.get_observation())
    # sleep(2)
    # print(pe.arm.set_state(1,0,0,0,0))
    # print(pe.get_observation())
    # sleep(2)
    # print(pe.get_observation())
    # print("Moved left")
    sleep(20)
    print(pe.get_observation())
    sleep(2)
    print(pe.step([1, 0]))
    print(pe.step([1, 0]))
    # print(pe.get_observation())
    sleep(2)
    print(pe.get_observation())
    print("Moved left")



