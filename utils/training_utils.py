from maze_env import MazeEnv_v0
from utils.PettingZooEnv_new import PettingZooEnv_new
from policy import DQNPolicy_new
import supersuit
import torch
import torch.nn as nn
import pickle

# define some helper functions for training and stuff

# single agent
def preprocess_maze_env(render_mode=None, size=6):
    env = MazeEnv_v0.env_single(render_mode=render_mode, size=size)
    env = supersuit.multiagent_wrappers.pad_observations_v0(env)
    env = PettingZooEnv_new(env)
    return env

def set_eps(policy, eps1, agents=None, eps2=None, single = True):
    """
    Set the exploration/exploitation parameter for the policy

    agents, eps2 and single=False are used for MARL (if implemented)
    """
    if single:
        policy.set_eps(eps1)
    else:
        # if MARL is implemented
        policy.policies[agents[0]].set_eps(eps1)
        policy.policies[agents[1]].set_eps(eps2)

# create a CNN for the observer
class CNN(nn.Module):
    def __init__(self, maze_width = 6, max_actions = 5):
        super().__init__()
        # size of cnn window after convolutions, change if required (if changing padding/stride/window)
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
        logits = self.model(obs)
        return logits, state

def watch(policy, collector, gym_reset_kwargs):
    """
    To render the policy and how it interacts with the environment
    """
    assert gym_reset_kwargs is not None, "Please input reset gym_reset_kwargs i.e. options"

    # set policy to eval mode
    policy.eval()
    collector.reset_env(gym_reset_kwargs=gym_reset_kwargs)
    #np.random.seed()
    collector.collect(n_episode=1, render=1/120, gym_reset_kwargs=gym_reset_kwargs)
    # reset back to training mode
    policy.train()

def save_model_buffer_ephist_abs(policy, replay_buffer, episode_history, filename, run_type):
    """
    Save the model and files after trinaing is completed
    """
    
    assert filename is not None, "Please insert a filename"
    # save the model
    filename_model = filename + "_model.pt"
    torch.save(policy.state_dict(), filename_model)

    # save the replay buffer
    filename_buffer = filename + "_buffer.pkl"
    with open(filename_buffer, "wb") as f:
        pickle.dump(replay_buffer, f)
    
    # save the successful episode history
    filename_ep_hist = filename + "_ephist.pkl"
    with open(filename_ep_hist, "wb") as f:
        pickle.dump(episode_history, f)

    # save abstractions
    if run_type == "abstraction":
        abstractions = policy.abstractions
        filename_abstractions = filename + "_abstractions.pkl"
        with open(filename_abstractions, "wb") as f:
            pickle.dump(abstractions, f)