# dm_control_cur Project Structure

## Overview

Please finish reading this document before contributing to the project.

## Project Structure

```
dm_control_cur
├── physical_arm_environment
│   ├── environments
│   └── utils
├── virtual_arm_environment
│   ├── environments
│   └── utils
├── utility_classes
│   ├── parameterizer.py
│   └── data_wrappers.py
├── ddpg
│   ├── ddpg_classes
│   ├── ddpg_unit_tests
│   ├── ddpg_residual_control
│   │   ├── data
│   │   ├── models
│   │   └── main.py
│   └── [...]
├── per
└── [...]
```

## Folder Descriptions

### virtual_arm_environment

This is our bootleg dm_control environment that houses the arm simulation.

### physical_arm_environment

This is the folder that houses the ```Firefly``` arm wrapper environment.

### utility_classes

The ```abstract_classes``` and ```simulator.py``` are located here. When you work on something new, i.e. a new
environment or model, you must inherit from the abstract classes here. If you do so, then you can plug it
into ```simulator.py``` and that should work.

This also contains other utility classes like ```parameterizer.py``` which is used to modify the object and robot
parameters like dimensions and position. We vary them so that the model can be trained to be robust to small changes in
the environment.

### ddpg (and others)

Each folder like ```ddpg``` contains various implementations of the reinforcement learning method, found
in ```ddpg_classes```. It also contains various folders where we train and store these models, such
as ```ddpg_residual_control```, which is ddpg added with residual learning.

If you modify the core components in ```ddpg_classes```, be sure to run the unit tests script to make sure that
everything is still working!

## Contribution Instructions

### Adding a new environment

For ```passive_hand.py```, you will need to follow how other control suite tasks are implemented, as the control suite
tasks do not inherit from a base class.

For the ```physical_arm_environment```, there is an abstract class in the ```utility_classes``` folder that contains the
methods that an environment object should implement.

Once you have written your classes, write tests in an adjacent folder (location isn't important) to check that the
environment's methods work properly.

### Adding a new agent

You should inherit from the abstract ```Agent``` class in ```utility_classes```.

Once done, write tests in an adjacent folder to check that the agent's methods work. This is important as it is
easy to break models when you modify internal parameters (lesson learnt from experience!)

### Running simulations

Instantiate the ```Simulation``` class located in ```simulator.py``` in ```utility_classes``` with the environment and
model that you want to work with. You can look at example code in the ```ddpg``` folder.

Write some tests to verify that the simulation works correctly. I have some example tests in ```ddpg/ddpg_unit_tests```.

