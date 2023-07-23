from typing import Any, Union
import numpy as np
from tianshou.data import Batch, to_numpy
from tianshou.policy.modelfree.dqn import DQNPolicy

class DQNPolicy_new(DQNPolicy):
    """
    A slightly modified policy from the Tianshou DQN policy, where we modify the class to allow 'custom' compound actions that are
    only determined during runtime from the abstractions.
    """

    def __init__(self, max_actions=5, *args, **kwargs):
        super(DQNPolicy_new, self).__init__(*args, **kwargs)

        if max_actions is None or max_actions < 5:
            max_actions = 5
            print("Minimum actions must be at least 5. Setting to 5")
        self.max_action_num = max_actions # no movement, up, down,. right, left for basic movement
        self.allow_abstractions = False

        if max_actions > 5: # only if more than 5 (i.e. we allow extra action spaces for the model)
            self.abstractions = {key:0 for key in range(5, max_actions)} # no abstractions will be represented by a 0 (no movement)
            self.allow_abstractions = True
        
        # action to direction dict (same as environment)
        self._action_to_direction = {
            0: np.array([0,0]), # no movement
            1: np.array([-1,0]), # up
            2: np.array([1,0]), # down
            3: np.array([0,1]), # right
            4: np.array([0,-1]) # left
        }

    def available_action_keys(self) -> list:
        """
        Return the action keys (5 to max_actions, or if available) that are currently not in use that can be used as keys for abstractions.
        """
        available_keys = []
        for key in range(5, self.max_action_num):
            if self.abstractions.get(key) == 0:
                available_keys.append(key) # append if no abstractions are used for the respective key
        
        return available_keys

    def used_action_keys(self):
        """
        Same as available_action_keys, but the opposite, where we list out the action keys used for abstractions
        """
        used_keys = []
        for key in range(5, self.max_action_num):
            if self.abstractions.get(key):
                used_keys.append(key) # append if no abstractions are used for the respective key
        
        return used_keys
            
    def add_abstraction(self, abstraction: np.ndarray) -> int:
        """
        Add the specified abstraction to the first available key. Return the key ID.
        """
        keys = self.available_action_keys()
        if len(keys) < 1:
            # i.e. there are no available keys
            print("Max actions reached. Cannot append any additional abstractions.")
        else:
            key = keys[0] # the first available key
            self.abstractions[key] = [abstraction] # the abstraction is saved as a list due to issues with the API and wrappers
            print(f"Appended abstraction to the agent. (Key: {key})")
            return key # return the key for the dreaming sequence so we can insert it into the buffer
    
    def remove_abstraction(self, key) -> None:
        """
        Remove the specified abstraction from the respective key.
        """
        # first, check if any other abstraction depends on the currently removed abstraction
        for abs_key in self.used_action_keys():
            if abs_key != key:
                if np.any(self.abstractions[abs_key][0] == key):
                    # if so, unnest it and make it its own abstraction with no abstraction dependencies
                    unnest_abs = self.unnest_abstractions(self.abstractions[abs_key][0])
                    self.abstractions[abs_key] = [unnest_abs]

        if self.abstractions.get(key):
            self.abstractions[key] = 0
        else:
            print("No abstractions found in this key")

    def map_action(self, act: np.ndarray) -> np.ndarray:
        """
        Map the action (if its 5 or more) to a respective abstraction (which will be a numpy array of primary actions (1 to 4))
        """
        if self.allow_abstractions and act[0] > 4:
            # if the number is for an abstraction and its allowed
            if self.abstractions.get(act[0]):
                # if an abstraction exists, it will return a list of len 1
                # unnest abstractions in case an abstraction contains another abstraction
                abstraction = self.unnest_abstractions(self.abstractions[act[0]][0])
                return [abstraction]
            else:
                print("Bad Key in Map Action, returning no movement")
                # otherwise do no movement as the action (and give warning)
                return np.array([0])
        else:
            # otherwise just return the action as it is
            return act
    
    def unnest_abstractions(self, abstraction):
        """
        For the cases where the abstractions have other abstractions themselves
        """
        abstraction_unnest = np.copy(abstraction)

        # make sure the actions given are primitives (i.e. in case an abstraction contains an abstraction)
        while np.all(abstraction_unnest < 5) is not True:
            idx = np.where(abstraction_unnest >= 5)[0] # the first idx
            if len(idx) == 0:
                break
            # put in the abstraction contents
            abstraction_unnest = np.concatenate([abstraction_unnest[:idx[0]], self.abstractions[abstraction_unnest[idx[0]]][0], abstraction_unnest[idx[0]+1:]])
        return abstraction_unnest

    
    def check_legal_moves_abstractions(self, obs: Batch, action_keys, abstraction_mask):
        """
        We cannot check the legal moves in the environment due to some limitations with how the wrappers work,
        So we have to check it in the policy itself.
        This is kinda a half-assed way to do it, but whatever
        """
        current_loc = np.where(obs.obs[0][1,:,:]==1)
        exit_loc = np.array(np.where(obs.obs[0][2,:,:]==1)).reshape(2)
        obs = obs.obs[0][0,:,:]

        # check the abstractions to see if they're valid
        for key in action_keys:
            invalid_movement = False
            at_exit = False
            abstraction = self.abstractions[key][0]

            # make sure the actions given are primitives (i.e. in case an abstraction contains an abstraction)
            abstraction = self.unnest_abstractions(abstraction)

            current_loc1 = np.copy(current_loc).reshape(2)

            # iterate through the actions one-by-one to see if if its valid
            for action in abstraction:
                direction = self._action_to_direction[action]
                # find new location
                new_loc = tuple(current_loc1 + direction)
                try: 
                    location = obs[new_loc]
                except:
                    # if failed, most likely because our of bounds fail this shit
                    abstraction_mask[key] = False
                    invalid_movement = True
                    break
                if int(location) == 1: # wall
                    abstraction_mask[key] = False
                    invalid_movement = True
                    break # if encouter wall, stop the abstraction validity search and set flag to true
                elif new_loc == tuple(exit_loc):
                    # check if it passes through exit (if the last action is the exit, it is ok)
                    at_exit = True
                    current_loc1 = new_loc
                elif at_exit:
                    # if it kept going, but at_exit flag was triggered, end the search (it passed through the exit so it's not valid)
                    abstraction_mask[key] = False
                    invalid_movement = True
                    break
                else:
                    current_loc1 = new_loc # otherwise keep going

            if invalid_movement:
                continue # go to next occupied abstraction key without changing the action mask
            else:
                # if no invalid_movement flag, the abstraction is valid.
                abstraction_mask[key] = True
        
        return abstraction_mask

    def check_legal_moves_abstractions_dream(self, obs: Batch, action_keys, abstraction_mask):
        """
        We cannot check the legal moves in the environment due to some limitations with how the wrappers work,
        So we have to check it in the policy itself.
        This is kinda a half-assed way to do it, but whatever
        """
        current_loc = np.where(obs.obs[1,:,:]==1)
        exit_loc = np.array(np.where(obs.obs[2,:,:]==1)).reshape(2)
        obs = obs.obs[0,:,:]

        # check the abstractions to see if they're valid
        for key in action_keys:
            invalid_movement = False
            at_exit = False
            abstraction = self.abstractions[key][0]

            # make sure the actions given are primitives (i.e. in case an abstraction contains an abstraction)
            abstraction = self.unnest_abstractions(abstraction)
            
            current_loc1 = np.copy(current_loc).reshape(2)

            # iterate through the actions one-by-one to see if if its valid
            for action in abstraction:
                direction = self._action_to_direction[action]
                # find new location
                new_loc = tuple(current_loc1 + direction)
                location = obs[new_loc]
                if int(location) == 1: # wall
                    abstraction_mask[key] = False
                    invalid_movement = True
                    break # if encouter wall, stop the abstraction validity search and set flag to true
                elif new_loc == tuple(exit_loc):
                    # check if it passes through exit (if the last action is the exit, it is ok)
                    at_exit = True
                    current_loc1 = new_loc
                elif at_exit:
                    # if it kept going, but at_exit flag was triggered, end the search (it passed through the exit so it's not valid)
                    abstraction_mask[key] = False
                    invalid_movement = True
                    break
                else:
                    current_loc1 = new_loc # otherwise keep going

            if invalid_movement:
                continue # go to next occupied abstraction key without changing the action mask (default false)
            else:
                # if no invalid_movement flag, the abstraction is valid.
                abstraction_mask[key] = True
        
        return abstraction_mask

    def abstraction_action_mask(self, obs: Batch):
        # check the size of the abstraction mask (to make sure we don't make it too large)
        if len(obs.mask[0]) == self.max_action_num:
            return

        # assume the mask is true/false boolean
        abstraction_mask = np.full(self.max_action_num, False) # make an array with all false
        keys = self.used_action_keys() # get the used actions keys (i.e. they have abstractions)

        # copy the current mask to the first 5 entries
        for i in range(len(obs.mask[0])):
            abstraction_mask[i] = obs.mask[0][i]

        # check the legal moves of the abstractions
        abstraction_mask = self.check_legal_moves_abstractions(obs, keys, abstraction_mask)

        # modify the original mask
        obs.mask = np.array([abstraction_mask])
    
    def abstraction_action_mask_dream(self, obs: Batch):
        """
        A slightly modified version of the abstraction_action_mask function above
        """
        # check the size of the abstraction mask (to make sure we don't make it too large)
        if len(obs.mask) == self.max_action_num:
            return

        # assume the mask is true/false boolean
        abstraction_mask = np.full(self.max_action_num, False) # make an array with all false
        keys = self.used_action_keys() # get the used actions keys (i.e. they have abstractions)

        # copy the current mask to the first 5 entries
        for i in range(len(obs.mask)):
            abstraction_mask[i] = obs.mask[i]

        # check the legal moves of the abstractions
        abstraction_mask = self.check_legal_moves_abstractions_dream(obs, keys, abstraction_mask)

        # modify the original mask
        obs.mask = abstraction_mask
        
    def forward(self, batch: Batch, state: Union[dict, Batch, np.ndarray] = None, model: str = "model", input: str = "obs", **kwargs: Any) -> Batch:
        """
        Mostly the same as Tianshou's DQNPolicy, but with minor modifications to the action masking to support abstractions (that cannot be directly passed in the environment)
        """
        model = getattr(self, model)
        obs = batch[input]

        # update the batch with new masks
        if self.allow_abstractions:
            self.abstraction_action_mask(obs)
        """
        print("after ----\n", batch)
        """
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        logits, hidden = model(obs_next, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        return Batch(logits=logits, act=act, state=hidden)