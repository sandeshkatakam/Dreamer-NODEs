import torch
import torch.utils.data as data
import torch.datasets as datasets
import torch.dataloader as dataloader

# Path: dataset/data_loader_utils.py


## MUJOCO Dataset and Data Loaders

## OPENAI GYM INTERFACE

class openai_gym_env():
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.env.reset()
        self.state = self.env.state
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def make_env(self):
        return self.env
    

import gymnasium
