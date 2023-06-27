from maze_env import MazeEnv_v0
from utils.PettingZooEnv_new import PettingZooEnv_new
import supersuit
import torch
import torch.nn as nn

# define some helper functions
# single agent
def preprocess_maze_env(render_mode=None, size=6):
    env = MazeEnv_v0.env_single(render_mode=render_mode, size=size)
    env = supersuit.multiagent_wrappers.pad_observations_v0(env)
    env = PettingZooEnv_new(env)
    return env

"""
def preprocess_maze_env_multi(render_mode=None, size=maze_width):
    env = MazeEnv_v0.env(render_mode=render_mode, size=size)
    env = supersuit.multiagent_wrappers.pad_observations_v0(env)
    env = PettingZooEnv_new(env)
    return env
"""
"""
def interleave_training(obs_train):
    if obs_train:
        policy.policies[agents[0]].set_eps(eps_train)
        policy.policies[agents[1]].set_eps(0)
        obs_train = obs_train != True
    else:
        policy.policies[agents[0]].set_eps(0)
        policy.policies[agents[1]].set_eps(eps_train)
        obs_train = obs_train != True
"""

def set_eps(policy, agents, eps1, eps2=None, single = False):
    if single:
        policy.set_eps(eps1)
    else:
        policy.policies[agents[0]].set_eps(eps1)
        policy.policies[agents[1]].set_eps(eps2)

# create a CNN for the observer
class CNN(nn.Module):
    def __init__(self, maze_width = 6, max_actions = 5):
        super().__init__()
        
        lin_size = ((((maze_width*2+1)-3+1)-3+1)-3+1)
        self.model = nn.Sequential(
            # assume maze size of 6x6 (13x13 with walls)
            nn.Conv2d(3, 16, 3), nn.ReLU(inplace=True),  # (13-3)+1 = 11, 
            nn.Conv2d(16, 32, 3), nn.ReLU(inplace=True), # 11-3+1=9, 
            nn.Conv2d(32, 64, 3), nn.ReLU(inplace=True), # 9-3+1=7
            nn.Flatten(), nn.Linear(64*lin_size*lin_size, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, max_actions)
        )
    
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        self.batch = obs.shape[0]
        #logits = self.model(obs.view(batch, -1))
        logits = self.model(obs)
        return logits, state

def watch(policy, collector, gym_reset_kwargs):
    assert gym_reset_kwargs is not None, "Please input reset kwargs i.e. options"
    # set policy to eval mode
    policy.eval()
    collector.reset_env(gym_reset_kwargs=gym_reset_kwargs)
    #np.random.seed()
    collector.collect(n_episode=1, render=1/120, gym_reset_kwargs=gym_reset_kwargs)
    # reset back to training mode
    policy.train()