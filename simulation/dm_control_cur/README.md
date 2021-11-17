# dm_control_cur Project Structure
## Overview
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
## virtual_arm_environment
This is our bootleg dm_control environment that houses the arm simulation.
## physical_arm_environment
This is the folder that houses the ```Firefly``` arm wrapper environment.
## utility_classes
The ```abstract_classes``` and ```simulator.py``` are located here. When you work on something new, i.e. a new environment or model, you must inherit from the abstract classes here. If you do so, then you can plug it into ```simulator.py``` and that should work.

This also contains other utility classes like ```parameterizer.py``` which is used to modify the object and robot parameters like dimensions and position. We vary them so that the model can be trained to be robust to small changes in the environment.
## ddpg
Each folder like ```ddpg``` contains various implementations of the reinforcement learning method, found in ```ddpg_classes```. It also contains various folders where we train and store these models, such as ```ddpg_residual_control```, which is ddpg added with residual learning.

If you modify the core components in ```ddpg_classes```, be sure to run the unit tests script to make sure that everything is still working!

