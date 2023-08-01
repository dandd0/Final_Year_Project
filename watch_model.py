# local imports
from utils.training_utils import *
from policy import DQNPolicy_new
import numpy as np
import tianshou as ts
import torch

# ---  RUN THE CODE ---
# set hyperparameters
lr = 5e-4 # the learning rate
gamma = 0.9 # gamma in dqn formula (nstep coefficient)
n_step = 3 # number of steps to look ahead
target_update_freq = 100 # number of update calls before updating target network
maze_width = 6 # maze width (not incl. walls)

# maze type
maze_type = input("Maze type: (random or structured) ") # random, trivial
assert maze_type in ["random", "structured"], "Only 'random' and 'structured maze supported."
total_mazes = 16 # total number of mazes

# ask for abstractions
run_type = input("What type of model? (baseline, abstraction or preabstract) ")
assert run_type in ["abstraction", "baseline", "preabstract"], "Only abstraction, baseline, and preabstract models supported."
if run_type == "abstraction":
    max_actions = 10
elif run_type == "preabstract":
    assert maze_type == "structured", "Preabstractions only for structured mazes"
    max_actions = 8
elif run_type == "baseline":
    max_actions = 5
else:
    print("broke")

print(f"Maze type: {maze_type}, Run type: {run_type}")

# set up training with no render environment
env_human = preprocess_maze_env(render_mode="human")
env_human = ts.env.DummyVectorEnv([lambda: env_human])

net_obs = CNN(maze_width=maze_width, max_actions=max_actions)
optim_obs = torch.optim.Adam(params=net_obs.parameters(), lr=lr)

# set up policy and collectors
agent_observer = DQNPolicy_new(max_actions, net_obs, optim_obs, gamma, n_step, target_update_freq)

policy = agent_observer

# load the models & abstractions
if run_type == "abstraction":
    if maze_type == "random":
        file_name = "random_abs"
    elif maze_type == "structured":
        file_name = "structured_abs"
elif run_type == "baseline":
    if maze_type == "random":
        file_name = "random_base"
    elif maze_type == "structured":
        file_name = "structured_base"
elif run_type == "preabstract":
    file_name = "structured_preabs"

full_file_name = "model_examples/"+file_name
if run_type == "abstraction":
    with open(full_file_name+"_abstractions.pkl", "rb") as f:
        abstractions = pickle.load(f)

policy.load_state_dict(torch.load(full_file_name+"_model.pt"))

if run_type == "preabstract":
    policy.add_abstraction(np.array([2,2,2,2,3,3,3,3]))
    print(f"\t{np.array([2,2,2,2,3,3,3,3])}")
    policy.add_abstraction(np.array([4,4,4,4,1,1,1,1]))
    print(f"\t{np.array([4,4,4,4,1,1,1,1])}")
    policy.add_abstraction(np.array([1,1,3,3,1,1,3,3]))
    print(f"\t{np.array([1,1,3,3,1,1,3,3])}")
elif run_type == "abstraction":
    abstractions = list(abstractions.values())
    for abstraction in abstractions:
        if isinstance(abstraction, list):
            policy.add_abstraction(np.array(abstraction[0]))
            print(f"\t{abstraction[0]}")

human_collector = ts.data.Collector(
    policy, 
    env_human, 
    exploration_noise=True
)

# set policy to eval mode
policy.set_eps(0)
policy.eval()
for seed in range(1, total_mazes):
    human_collector.reset_env(gym_reset_kwargs={"options":{"maze_type":maze_type, "n_mazes":seed, "random":False}})
    human_collector.collect(n_episode=1, render=1/10, gym_reset_kwargs={"options":{"maze_type":maze_type, "n_mazes":seed, "random":False}})
