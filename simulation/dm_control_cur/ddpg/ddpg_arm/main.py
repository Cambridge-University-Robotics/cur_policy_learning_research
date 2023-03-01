from ddpg_classes.simulator_arm import SimulationArm
import simulation.dm_control_cur.virtual_arm_environment.environments as env


def fix_xml(
        path_to_xml: str = r'..\..\virtual_arm_environment\environments\assets\passive_hand\lift.xml'):
    with open(path_to_xml, 'r') as f:
        content = f.read()

    compiler_options_str = content.split('<compiler')[1].split('/>')[0]
    comp_opt_position = content.find('<compiler')
    rel_end_of_comp_opt_position = content.split('<compiler')[1].find('/>')
    option = ' autolimits="true"'
    if compiler_options_str.find(option) < 0:
        compiler_options_str += option
        compiler_options_str += '/>'
        new_content = content[:comp_opt_position+len('<compiler')] + compiler_options_str + content[comp_opt_position + len('<compiler') + rel_end_of_comp_opt_position + 2:]

        with open(path_to_xml, 'w') as f:
            f.write(new_content)


fix_xml()

sa = SimulationArm(
    load_model=False,
    plot=False,
    name_model='passive_hand',
    task='lift_sparse',
    label=None,  # tag that is appended to file name for models and graphs
    num_episodes=1000,  # number of simulation rounds before training session
    batch_size=128,  # number of past simulations to use for training
    duration=200,  # duration of simulation

    env=env,  # used if we inject an environment
    dim_obs=33,
)

sa.update_xml(obj_amt=0, rbt_amt=0)
# sa.train()
sa.show_simulation()
