# dm_control_cur Project Structure
## Overview
```
dm_control_cur/
├── simulation_control
│   ├── environments
│   └── utils
├── utility_classes
│   ├── parameterizer.py
│   └── data_wrappers.py
├── ddpg
│   ├── ddpg_classes
│   ├── ddpg_residual_control
│   │   ├── data
│   │   ├── models
│   │   └── main.py
│   └── [...]
├── per
└── [...]
```
## simulation_control
This is our bootleg dm_control environment that houses the arm simulation.
## utility_classes
This contains ```parameterizer.py``` which is used to modify the object and robot parameters like dimensions and position. We vary them so that the model can be trained to be robust to small changes in the environment.
## ddpg
Each folder like ```ddpg``` contains various implementations of the reinforcement learning method, found in ```ddpg_classes```. It also contains various folders where we train and store these models, such as ```ddpg_residual_control```, which is ddpg added with residual learning.

