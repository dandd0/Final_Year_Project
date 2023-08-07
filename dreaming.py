import numpy as np
import torch
import pickle

from policy import DQNPolicy_new
from maze_env.MazeEnv_v0 import *
from collections import deque
from tianshou.data import VectorReplayBuffer, ReplayBuffer, Batch

def is_sub_arr_np(a1, a2):
    """
    To check whether the abstraction is a subarray of the episode action history

    Credits to: https://stackoverflow.com/questions/57004175/numpy-check-if-1-d-array-is-sub-array-of-another
    """
    l1, = a1.shape
    s1, = a1.strides
    l2, = a2.shape
    # make sure array 1 is longer than array 2
    if l2 > l1:
        return False, None # any_sub, indices

    a1_win = np.lib.stride_tricks.as_strided(a1, (l1 - l2 + 1, l2), (s1, s1))
    any_sub = np.any(np.all(a1_win == a2, axis=1))

    if any_sub:
        indices = np.where(np.all(a1_win == a2, axis=1)==True)[0][0] # return the first indices where it matches
        return any_sub, [indices, indices+l2]
    else:
        return any_sub, None
    
def replace_new_action(action_history, abstraction_key, indices):
    """
    Replace the indices specified (a tuple with start and end idxs) with the new abstraction key

    action history: the successful episode history's action array
    abstraction_key: the key representation of the new abstraction for the agent
    """
    return np.concatenate([action_history[:indices[0]], np.array([abstraction_key]), action_history[indices[1]:]])

def new_episode_actions(episode_history, abstraction, abstraction_key):
    """
    To get the new episode history with the abstraction replaced with the new key
    """

    new_actions = []
    for i in range(len(episode_history)):
        # get the episode actions
        ep_act = episode_history[i].act
        
        # set some flags
        is_sub = True
        any_change = False
        
        # keep replacing the abstractions until no more sublist exists
        while is_sub:
            is_sub, indices = is_sub_arr_np(ep_act, abstraction) # check if a sublist exists and their indices
            if is_sub:
                # if sublist exists, replace the indices area with the abstraction key (for the buffer)
                ep_act = replace_new_action(ep_act, abstraction_key, indices)
                any_change = True

        # only make note of episodes with the abstraction in it
        if any_change:
            ep_dict = {"ep_number":i, "actions":ep_act}
            new_actions.append(ep_dict)

    return new_actions

def new_sample_with_abstraction(policy: DQNPolicy_new, env: MazeEnv_single, maze_type: str, new_actions: np.ndarray, maze_seed: int, abstraction_buffer: ReplayBuffer):
    """
    Interact with the environment with the new action list and add it to the abstraction buffer
    """
    data = Batch(obs={}, act={}, rew={}, terminated={}, truncated={}, done={}, obs_next={}, info={}, policy={})
    # reset the environment (with the correct seed)
    obs, info = env.reset(options={"maze_type":maze_type, "n_mazes":maze_seed, "random":False})
    # initial observations
    data.obs = obs
    data.info = info

    # loop through defined actions
    for action in new_actions:
        # update batch with action and policy
        data.update(act=action)

        # modify the mask (for abstraction buffer appending purposes later)
        policy.abstraction_action_mask_dream(data.obs)

        # remap action and interact with environment
        action_remap = policy.map_action(np.array([action])) # get the remapped action (e.g. abstraction from action number 5)
        obs_next, rew, terminated, truncated, info = env.step(action_remap)
        done = np.logical_or(terminated, truncated)

        # update batch (to send to buffer later)
        data.update(obs_next=obs_next, rew=rew, terminated=terminated, truncated=truncated, done=done, info=info)
        
        # add data into the buffer
        abstraction_buffer.add(data)

        # set obs to be obs_next
        data.obs = obs_next
        
def dream(policy: DQNPolicy_new, abstraction: np.ndarray, env: MazeEnv_single, episode_history: deque, maze_type: str):
    """
    The dreaming function for reinforcing the new abstraction knowledge to the model.
    """
    # make sure there is at least an abstraction
    if len(abstraction) == 0:
        return
    
    # insert the new abstraction into the policy
    abstraction_key = policy.add_abstraction(abstraction)
    new_actions = new_episode_actions(episode_history, abstraction, abstraction_key)

    # make sure there are new actions with the abstraction (really shouldnt be possible but its safer to do this)
    if len(new_actions) == 0:
        return
    
    # set up the buffer that will be used for abstraction learning
    abstraction_buffer = ReplayBuffer(100000) # some large number
    print(f"Resampling environment steps with new abstractions... ({len(new_actions)} episodes with abstractions found.)")

    for count, actions_dict in enumerate(new_actions):
        # get information out
        ep_number = actions_dict["ep_number"]
        actions = actions_dict["actions"]
        if maze_type == "random":
            maze_seed = episode_history[ep_number]['info']['nth_maze'][0] # only the case for random
        else:
            maze_seed = episode_history[ep_number]['info']['maze_seed'][0] # for both trivial and structured
            

        # will update the abstraction buffer by reference (for each respective episode)
        new_sample_with_abstraction(policy, env, maze_type, actions, maze_seed, abstraction_buffer)

    # update the policy with new buffer (the entire buffer)
    policy.update(0, abstraction_buffer)
    print(f"\nAgent has learned the new abstraction: {abstraction}")

    return abstraction_buffer

def find_bad_abstractions(policy: DQNPolicy_new, episode_history: deque, eps: int):
    """
    To find and remove abstractions that are used less than random exploration.
    """
    # if there are no abstractions
    if len(policy.used_action_keys()) == 0:
        return

    print("Finding bad abstractions...")

    # take the episode history and use the actions list as our data
    # # we can set the threshold to be: random search factor * number of steps / ((total number of available moves) * (len of abstraction))

    actions = []
    for i in range(len(episode_history)):
        for j in range(len(episode_history[i].act)):
            actions.append(episode_history[i].act[j])
    actions = np.array(actions)

    num_abstractions = len(policy.used_action_keys())

    # get the count for the abstraction actions
    abstraction_actions = actions[actions >= 5]
    action, count = np.unique(abstraction_actions, return_counts=True)

    for a, c in zip(action, count):
        abstraction_len = len(policy.unnest_abstractions(policy.abstractions[a][0]))
        threshold = (eps*len(actions)) / ((5+num_abstractions)*abstraction_len)
        print(f"\tAbstraction uses: {a} = {c}, {policy.abstractions[a][0]}")
        print(f"\t\tMin # for abstraction {a}: {threshold}")
        if c < threshold:
            # remove the abstraction if count is below threshold
            print(f"\tRemoving abstraction: {a}, {policy.abstractions[a][0]}")
            policy.remove_abstraction(a)
            return # end search early, only remove 1 at a time



    