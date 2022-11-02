# Cambridge University Robotics Policy Learning Research

## Introduction

This is CUR's research project repository. We are currently working on a reinforcement learning project, where we use
residual and transfer learning to train a physical ```Firefly``` arm.

## Project Structure

```
cur_policy_learning_research
└── simulation
    └── dm_control_cur
```
# Repo
## Setup
### Cloning the Repository
Figure it out! There are a lot of resources online about how to setup a github repository.

```git clone git@github.com:Cambridge-University-Robotics/cur_policy_learning_research.git```
### Create a Virtual Environment
We are using Python 3.9.7.

Managing Python installations is complex. The way to simplify it is that you can have multiple global installations of different versions using a tool like ```pyenv``` or your IDE like Pycharm, and then multiple virtual environments of a version using the module ```venv```. The steps are:
#### Install ```pyenv```/Using IDE
Windows: ```https://github.com/pyenv-win/pyenv-win```

Linux/Mac: ```https://github.com/pyenv/pyenv```

IDE: ```https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html#add_new_project_interpreter```

This allows you to install multiple versions of Python, which turns out to be really useful. 
#### Creating a Virtual Environment
A virtual environment is a copy of your current global Python, which might have more packages installed. You traditionally it with ```python -m venv whatevernameyouwant```, but IDEs can help you to create them automatically.
#### Installing the Stuff
We need some packages for our code to work.

```pip install -r requirements.txt```
### Running Some Code
There are many ways you can run some code. Running from the command line would be something like (being in the base project directory):

```python -m simulation.dm_control_cur.ddpg.ddpg_humanoid.main```

To run the code in 
https://github.com/Cambridge-University-Robotics/cur_policy_learning_research/tree/master/simulation/dm_control_cur/ddpg/ddpg_humanoid

## Descriptions
The main thing that you want to be looking at is ```/simulation/dm_control_cur/ddpg/ddpg_classes``` and ```/simulation/dm_control_cur/ddpg/ddpg_humanoid```. The first folder contains a ddpg model, and the second folder contains code you can run to use the model. You can try and trace the execution of the code in order to understand how you are supposed to use the model to do predictions. However, you will realise that the code is very complicated, so feel free to rip the model code out and use it by itself.
