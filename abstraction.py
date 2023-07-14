import numpy as np
from collections import Counter
from policy import DQNPolicy_new

def longest_common_substring(s1, s2):
   """
   Where s1 is the longer string, and s2 is the shorter string

   Credits to https://en.wikibooks.org/w/index.php?title=Algorithm_Implementation/Strings/Longest_common_substring#Python
   """
   m = [[0] * (1 + len(s2)) for i in range (1 + len(s1))]
   longest, x_longest = 0, 0
   for x in range(1, 1 + len(s1)):
       for y in range(1, 1 + len(s2)):
           if s1[x - 1] == s2[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
           else:
               m[x][y] = 0
   return s1[x_longest - longest: x_longest]

def arr_to_string(actions):
    """
    Convert the actions array from the successful episode into a string for LCS
    """
    return "".join(actions.astype(str))

def string_to_arr(abstraction):
    """
    Convert the actions string into an array
    """
    return np.array(list(abstraction), dtype=int)

def find_abstractions(episode_history):
    """
    Find the best abstractions between all successful episode history pairs
    """

    LCS_counter = Counter()

    for i in range(len(episode_history)-1):
        for j in range(i+1, len(episode_history)):
            ep0 = arr_to_string(episode_history[i].act.astype(str))
            ep1 = arr_to_string(episode_history[j].act.astype(str))

            # find longest common substring
            if len(ep0) > len(ep1):
                LCS = longest_common_substring(ep0, ep1)
            else:
                LCS = longest_common_substring(ep0, ep1)

            # only add if there's any common substrings
            LCS_counter[LCS] += 1

    return LCS_counter

def abstraction_rating(LCS_counter):
    """
    get the ratings where its:
    (length of abstraction) * (number of times it occurred)
    """
    return [len(candidate) * amount for candidate, amount in LCS_counter.items()] 
    
def get_abstraction(LCS_counter,):
    """
    Find the best abstraction using the ratings
    """
    ratings = abstraction_rating(LCS_counter)
    best_rating = np.argmax(ratings)
    abstraction = list(LCS_counter.keys())[best_rating]
    # check if any abstractions found (or at least greater than 1 action)
    if len(abstraction) <= 1:
        return []

    # convert the string abstraction representation to an array
    return abstraction

def generate_abstractions(policy: DQNPolicy_new, episode_history):
    """
    The main function to call for generating abstractions
    """
    # check if the number of abstractions has already reached the limit
    if len(policy.available_action_keys()) == 0:
        print("Max actions reached. Finding bad abstractions instead.")
        return []

    # find the potential abstractions:
    LCS_counter = find_abstractions(episode_history)

    # return none if nothing found
    if len(LCS_counter.items()) <= 1:
        print("no common strings found, fix here")
        return []

    # get the abstraction
    abstraction = get_abstraction(LCS_counter)
    print(f"Best abstraction found: {abstraction}")
    abstraction = string_to_arr(abstraction)

    # make sure the abstraction found doesn't already exist
    for key, value in policy.abstractions.items():
        # only check if an abstraction exists
        if isinstance(value, list):
            # check if abstraction already exists
            if np.array_equal(abstraction, value[0]):
                print("Abstraction already exists.")
                return []
    
    return abstraction

