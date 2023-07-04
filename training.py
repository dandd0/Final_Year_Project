# local imports
from utils.training_utils import *
from policy import DQNPolicy_new
from abstraction import generate_abstractions
from dreaming import dream, find_bad_abstractions

# library imports
import tianshou as ts
import numpy as np
from collections import deque
import wandb

def log_data_wandb(result, eps_train, episodes_total, steps_total, train=True):
    """
    Pass a dict of data into the logger for WandB
    """
    # assume that at least 1 episode has been completed
    if train:
        # log the training data
        log_data = {"train":{
                "episode": result["n/ep"],
                "obs_reward": np.mean(result["rews"]),
                "length": result["len"],
                "exploration rate": eps_train,
                "episodes": episodes_total
            }
        }
    else:
        # otherwise assume it is testing data
        log_data = {"test":{
                "obs_reward": np.mean(result["rews"]),
                "length": result["len"],
                "obs_reward_std": np.std(result["rews"]),
            }
        }
    # log the data to WandB
    wandb.log(data=log_data, step=steps_total)

def test_policy(policy: DQNPolicy_new, mazes: int, test_collector: ts.data.Collector):
    """
    Test the agent with the current maze and all previous mazes.
    Return the results (for logging), the test mazes progress (for human monitoring) and booleans indicating successes
    """
    set_eps(policy, eps_test, single=True)
    policy.eval()
    
    test_mazes = []
    for seed in range(1, mazes+1):
        # test through all previous mazes
        test_collector.reset_env(gym_reset_kwargs={"options":{"maze_type":maze_type, "n_mazes":seed, "random":False}})
        test_result = test_collector.collect(n_episode=test_num, gym_reset_kwargs={"options":{"maze_type":maze_type, "n_mazes":seed, "random":False}})
        
        # if any of the previous mazes failed, break out of loop and do the train loop again
        if np.mean(test_result['rews']) < threshold_rew:
            passed_mazes = False
            passed_before = False
            test_mazes.append(0)
            return test_result, test_mazes, seed, False
        else:
            test_mazes.append(1)
    
    return test_result, test_mazes, seed, True

def train_loop(policy: DQNPolicy_new, # the policy itself
               agents, # a list of agents (in case i still want to do marl)
               train_collector: ts.data.Collector, # collectors
               test_collector: ts.data.Collector, 
               human_collector: ts.data.Collector,
               episode_history: deque, # for abstractions
               env: MazeEnv_v0 # for abstractions
               ):
    
    # manual training loop
    np.random.seed()
    passed_mazes = True # a flag for whether the policy passed the mazes
    steps_total = 0 # steps count total
    episodes_total = 0 # total number of episodes so far
    high_eps_run = False # for random high eps run

    for mazes in range(1, total_mazes):
        steps_within_maze = 0 # for counting the number of steps so far within a new introduction of a maze
        recent_abstraction = False # so we don't constantly check for abstractions
        recent_abstraction_timer = 0 # a timer for a grace period between an introduction of an abstraction and the policy's ability to remove it
        
        if passed_mazes == True:
            # reset epsilon again for the new maze
            eps_train = 0.9
            passed_mazes = False
            print(f"Current number of mazes: {mazes}")
        else:
            print("Failed to find a solution within suitable time. Stopping training.")
            return False # failed

        # reset
        set_eps(policy, eps_train, single=True)
        policy.train()
        
        for epoch in range(epochs):
            # reset number of steps couhnt
            steps_n = 0 
            
            # training loop
            # --- WAKE ---
            while steps_n < step_per_epoch:
                # have runs where the exploration rate is very high
                if np.random.randint(0, 10) == 0:
                    eps_prev = eps_train
                    eps_train = 0.95
                    high_eps_run = True
                
                set_eps(policy, eps_train, single=True)

                # train the model in training environment
                train_collector.reset_env(gym_reset_kwargs={"options":{"maze_type":maze_type, "n_mazes":mazes, "random":True}})
                result = train_collector.collect(n_episode=ep_per_collect, gym_reset_kwargs={"options":{"maze_type":maze_type, "n_mazes":mazes, "random":True}})
                steps_n += int(result['n/st'])
                steps_within_maze += int(result['n/st'])
                steps_total += int(result['n/st'])
                episodes_total += int(result['n/ep'])

                # add the successful episode steps into the episode history
                if result['rews'] > threshold_rew:
                    episode_history.append(train_collector.buffer[int(result['idxs']):int(result['idxs'] + result['lens'])])
                
                # update the parameters after each ep (online model)
                policy.update(batch_size, train_collector.buffer)

                #  reset high exploration
                if high_eps_run:
                    eps_train = eps_prev
                    high_eps_run = False

                # set the random training epsilon after each steps per collect
                # decay it by specified parameter every
                eps_train *= eps_decay
                eps_train = np.max([eps_train, eps_min])
                
                # log training data
                log_data_wandb(result, eps_train, episodes_total, steps_total, True)
            
            # check test results
            passed_before = True
            test_result, test_mazes, seed, test_flag = test_policy(policy, mazes, test_collector)

            if test_flag == False:
                passed_before = False
                passed_mazes = False

            # log testing data
            log_data_wandb(test_result, eps_train, episodes_total, steps_total, False)
            
            print(f"Evaluation Reward at Epoch {epoch+1}. Obs: {np.round(np.mean(test_result['rews']), 3)}, Maze: {seed}")
            print(f"Test Mazes results: {test_mazes}")

            # every n epochs render the policy for human-based evalution
            if (epoch % 10) == 0:
                for seed in range(1, mazes+1):
                    watch(policy, human_collector, {"options":{"maze_type":maze_type, "n_mazes":seed, "random":False}})

            # check if the agent can auccessfully solve the maze (within some threshold)
            if passed_mazes:
                print(f"Agents solved the current maze and all previous mazes. Solved all mazes on epoch {epoch+1}.")

                # watch the results if successful in passing twice
                for seed in range(1, mazes+1):
                    watch(policy, human_collector, {"options":{"maze_type":maze_type, "n_mazes":seed, "random":False}})
                break
            
            # to see if the agent can do it consequtively
            if passed_before:
                # pass the test maze at least twice
                passed_mazes = True
                print("Passed once")
            
            if run_type == "abstraction": # only do the sleep phase if abstractions are allowed
                # --- SLEEP ---
                if (steps_within_maze > 100000) and (mazes >= 3) and (recent_abstraction == False):
                    # minimum of 3 mazes before we start looking for abstractions
                    # if the policy is unable to find a solution after 10 evaluations, start finding abstractions
                    abstraction = generate_abstractions(policy, episode_history)
                    # --- DREAM ---
                    if len(abstraction) > 0:
                        # if an abstraciton is found, learn it
                        abstraction_buffer = dream(policy, abstraction, env, episode_history, maze_type) # the model is updated within here
                        train_collector.buffer.update(abstraction_buffer) # append new episodes to the end to the buffer (for future updating)

                        recent_abstraction = True # only allow 1 abstraction per new maze introduction (to prevent constant searching/instant deletion)
                        recent_abstraction_timer = 10 # a 'grace period' between the introduction of a new abtraction and the ability to remove it
                    elif recent_abstraction_timer == 0:
                        # else find and remove bad abstractions
                        find_bad_abstractions(policy, train_collector.buffer, eps_train)
                    else:
                        recent_abstraction_timer -= 1

            # reset back to training
            set_eps(policy, eps_train, single=True)
            policy.train()

    # end training and logging, return successful bool if it finishes properly
    print('Finished Training.')
    return True

def run_train_test_loop():
    """
    Set up all of the collectors, agents and other various things before training
    """

    # get the vectorized training/testing environments
    train_envs = ts.env.DummyVectorEnv([preprocess_maze_env for _ in range(train_num)])
    test_envs = ts.env.DummyVectorEnv([preprocess_maze_env for _ in range(test_num)])

    # set up training with no render environment
    env = preprocess_maze_env()

    # set up human render environment
    env_human = preprocess_maze_env(render_mode="human")
    env_human = ts.env.DummyVectorEnv([lambda: env_human])

    # get agent names
    agents = env.agents

    # observation spaces/action spaces for the two agents
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    net_obs = CNN(maze_width=maze_width, max_actions=max_actions)
    optim_obs = torch.optim.Adam(params=net_obs.parameters(), lr=lr)

    # set up policy and collectors
    agent_observer = DQNPolicy_new(max_actions, net_obs, optim_obs, gamma, n_step, target_update_freq)

    policy = agent_observer

    # define the training collector (the calc q and step functions)
    train_collector = ts.data.Collector(
        policy, 
        train_envs, 
        ts.data.VectorReplayBuffer(buffer_size, train_num),
        exploration_noise=True
    )

    # define the testing collector
    test_collector = ts.data.Collector(
        policy, 
        test_envs,
        exploration_noise=True
    )

    human_collector = ts.data.Collector(
        policy, 
        env_human, 
        exploration_noise=True
        )
    
    # for abstractions
    episode_history = deque(maxlen=500) # 500 most recent successful episodes

    # train
    try:
        success = train_loop(policy, agents, train_collector, test_collector, human_collector, episode_history, env)
        if success:
            # if finished everything
            wandb.finish(0) # successful run
            file_name = "model/" + str(maze_type) + "_" + str(run_type) + "_" + str(file_suf) + "_SUCCESS"
            save_model_buffer_ephist_abs(policy, train_collector.buffer, episode_history, file_name, run_type)
        else:
            # if stalled
            wandb.finish(1) # failed run
            file_name = "model/" + str(maze_type) + "_" + str(run_type) + "_" + str(file_suf) + "_FAILED"
            save_model_buffer_ephist_abs(policy, train_collector.buffer, episode_history, file_name, run_type)
    
    except KeyboardInterrupt:
        # if interrupted
        wandb.finish(2) # interrupted run (failed)
        print("RUN INTERRUPTED")
        file_name = "model/" + str(maze_type) + "_" + str(run_type) + "_" + str(file_suf) + "_INTERRUPT"
        save_model_buffer_ephist_abs(policy, train_collector.buffer, episode_history, file_name, run_type)

# ---  RUN THE CODE ---
# set hyperparameters
eps_train = 0.95 # exploration rate for training
eps_test = 0.0 # exploration rate for testing 
eps_decay = 0.999 # the exploration rate decay
eps_min = 0.15 # the minimum exploration rate
lr = 5e-4 # the learning rate
epochs  = 150 # max epochs per new maze introduction
batch_size = 512 # the update batch size
gamma = 0.9 # gamma in dqn formula (nstep coefficient)
n_step = 3 # number of steps to look ahead
target_update_freq = 100 # number of update calls before updating target network
train_num = 10 # num of simultaneous training environments (i think with n_episode=1, this doesn't matter but whatever)
test_num = 1 # num of simultaneous testing environments 
buffer_size = 30000 # buffer size
step_per_epoch = 10000 # number of steps for each epoch
step_per_collect = 200 # number of steps to collect before updating
ep_per_collect = 1 # number of episodes before updating
maze_width = 6 # maze width (not incl. walls)
n_mazes = 0 # the (initial) number of mazes
total_mazes = 2 # total number of random mazes ---- for the trivial maze, we use 36 (since it should be 'easier')
threshold_rew = 0.9 # threshold reward to consider a maze passed

# file name suffix
file_suf = input("File suffix (e.g. Date + run number): ") # the file name suffix (so it can be uniquely named)

# maze type
maze_type = input("Maze type: ") # random, trivial
assert maze_type in ["random", "trivial"], "Only 'random' and 'trivial' maze supported."

# max number of actions and abstractions
max_actions = int(input("Maximum number of actions (5 for Baseline, 5> for Abstraction): ")) # the maximum number of actions allowed for the policy (default should be at least 5)
if max_actions > 5:
    run_type = "abstraction"
elif max_actions == 5:
    run_type = "baseline"
else:
    print("Max actions must be at least 5. Defaulting to baseline model.")
    max_actions = 5
    run_type = "baseline"

# logger initialization
wandb.login()
run = wandb.init(
    project="Final_Year_Project",
    config={
        "eps_train":eps_train, "eps_test":eps_test,
        "eps_decay": eps_decay, "eps_min":eps_min,
        "learning rate": lr, "epochs": epochs, "batch_size":batch_size,
        "gamma": gamma, "n_step":n_step, "target_update_freq":target_update_freq,
        "buffer_size":buffer_size,
        "step_per_epoch":step_per_epoch, "step_per_collect":step_per_collect, "ep_per_collect":ep_per_collect,
        "maze width": maze_width, "n_mazes":n_mazes, "total_mazes":total_mazes,
        "threshold_rew":threshold_rew, "maze_type":maze_type,
        "non-marl":True, "max_actions":max_actions
    }
)

run_train_test_loop()