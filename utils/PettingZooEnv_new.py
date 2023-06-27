# THIS IS ALMOST THE SAME AS TIANSHOU'S BUILT IN PETTINGZOOENV WRAPPER, WITH MINOR MODIFICATIONS
# Instead of accessing the action mask in observation['action_mask']
# it will access it in info['action_mask']

import warnings
from abc import ABC
from typing import Any, Dict, List, Tuple

import pettingzoo
from gymnasium import spaces
from packaging import version
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper
from tianshou.env.pettingzoo_env import PettingZooEnv

if version.parse(pettingzoo.__version__) < version.parse("1.21.0"):
    warnings.warn(
        f"You are using PettingZoo {pettingzoo.__version__}. "
        f"Future tianshou versions may not support PettingZoo<1.21.0. "
        f"Consider upgrading your PettingZoo version.", DeprecationWarning
    )

class PettingZooEnv_new(PettingZooEnv):
    """
    Inherit the PettingZooEnv class from Tianshou.env.pettingzoo_env

    We need to inherit it due to a strange interaction in the DummyVectorEnv call, where it believes that this is not a gym environment?
    Somehow by doing this it works but apparently not if you copy and paste the class with minor modifications.
    """
    def __init__(self, env: BaseWrapper):
        super().__init__(env)
    
    def reset(self, *args: Any, **kwargs: Any) -> Tuple[dict, dict]:
        self.env.reset(*args, **kwargs)

        observation, reward, terminated, truncated, info = self.env.last(self)

        if isinstance(observation, dict) and 'action_mask' in observation:
            observation_dict = {
                'agent_id': self.env.agent_selection,
                'obs': observation['observation'],
                'mask':
                [True if obm == 1 else False for obm in observation['action_mask']]
            }
        elif 'action_mask' in info:
            observation_dict = {
                'agent_id': self.env.agent_selection,
                'obs': observation,
                'mask':
                [True if obm == 1 else False for obm in info['action_mask']]
            }
        else:
            if isinstance(self.action_space, spaces.Discrete):
                observation_dict = {
                    'agent_id': self.env.agent_selection,
                    'obs': observation,
                    'mask': [True] * self.env.action_space(self.env.agent_selection).n
                }
            else:
                observation_dict = {
                    'agent_id': self.env.agent_selection,
                    'obs': observation,
                }

        return observation_dict, {}
    
    def step(self, action: Any) -> Tuple[Dict, List[int], bool, bool, Dict]:
        self.env.step(action)

        observation, rew, term, trunc, info = self.env.last()

        if isinstance(observation, dict) and 'action_mask' in observation:
            obs = {
                'agent_id': self.env.agent_selection,
                'obs': observation['observation'],
                'mask':
                [True if obm == 1 else False for obm in observation['action_mask']]
            }
        elif 'action_mask' in info:
            obs = {
                'agent_id': self.env.agent_selection,
                'obs': observation,
                'mask':
                [True if obm == 1 else False for obm in info['action_mask']]
            }
        else:
            if isinstance(self.action_space, spaces.Discrete):
                obs = {
                    'agent_id': self.env.agent_selection,
                    'obs': observation,
                    'mask': [True] * self.env.action_space(self.env.agent_selection).n
                }
            else:
                obs = {'agent_id': self.env.agent_selection, 'obs': observation}

        for agent_id, reward in self.env.rewards.items():
            self.rewards[self.agent_idx[agent_id]] = reward
        return obs, self.rewards, term, trunc, {}
