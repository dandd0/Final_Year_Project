from utils.training_utils import *

import tianshou as ts

from collections import deque
import numpy as np
import wandb

# set hyperparameters
eps_train, eps_test = 0.95, 0.0 # exploration rate for training and testing respectively
eps_decay, eps_min = 0.999, 0.15 # the exploration rate decay and the minimum exploration rate
lr, epochs, batch_size = 5e-4, 150, 512 # the learning rate, max epochs per new maze intro and the update batch size
gamma, n_step, target_update_freq = 0.9, 3, 100 # gamma in dqn formula, number of steps to look ahead, number of update calls before updating target network
train_num, test_num = 10, 1 # num of simultaneous training and testing environments respectively
buffer_size = 30000 # buffer size
step_per_epoch, step_per_collect, ep_per_collect = 10000, 200, 1 # number of steps for each epoch, number of steps to collect before updating, number of episodes before updating
maze_width = 6 # maze width (not incl. walls)
high_eps_run, obs_train, passed_mazes = False, True, True # for random high eps run, for interleaving (might be broken?), whether the policies passed the mazes
steps_total, steps_n, episodes_total = 0, 0, 0 # steps count total, steps count within epoch, total number of episodes so far
n_mazes, total_mazes = 0, 16 # start with 3 (it will add one later) mazes initially to prevent single maze overfitting, total number of random mazes
# for the trivial maze, we use 36 (since it should be 'easier')
test_mazes = [] # for printing later
threshold_rew = 0.5 # threshold reward to consider a maze passed (tentative value)
maze_type = "random" # the type of maze to pass into the environment

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
        "non-marl":True
    }
)

def run():
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

    net_obs = CNN(maze_width=maze_width)
    optim_obs = torch.optim.Adam(params=net_obs.parameters(), lr=lr)

    # set up policy and collectors
    agent_observer = ts.policy.DQNPolicy(net_obs, optim_obs, gamma, n_step, target_update_freq)

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

    human_collector = ts.data.Collector(policy, env_human, exploration_noise=True)

    abstraction_buffer = ts.data.ReplayBuffer(30000) # some large number, this will be reset anyways
    episode_history = deque(maxlen=500) # 500 most recent successful episodes

    # manual training loop
    np.random.seed()
    for mazes in range(1, total_mazes):
        if passed_mazes == True:
            # reset epsilon again for the new maze
            eps_train = 0.9
            passed_mazes = False
            print(f"Current number of mazes: {mazes}")
        else:
            print("Failed to find a solution within suitable time. Stopping training.")
            break

        # reset
        set_eps(eps_train, single=True)
        policy.train()
        
        for epoch in range(epochs):
            # reset number of steps couhnt
            steps_n = 0 
            
            # training loop
            while steps_n < step_per_epoch:
                # have runs where the exploration rate is very high
                if np.random.randint(0, 10) == 0:
                    eps_prev = eps_train
                    eps_train = 0.95
                    high_eps_run = True
                
                set_eps(eps_train, single=True)

                # train the model in training environment
                train_collector.reset_env(gym_reset_kwargs={"options":{"maze_type":maze_type, "n_mazes":mazes, "random":True}})
                result = train_collector.collect(n_episode=ep_per_collect, gym_reset_kwargs={"options":{"maze_type":maze_type, "n_mazes":mazes, "random":True}})
                steps_n += int(result['n/st'])
                steps_total += int(result['n/st'])
                episodes_total += int(result['n/ep'])
                
                # update the parameters after each ep
                policy.update(batch_size, train_collector.buffer)

                #  reset high exploration
                if high_eps_run:
                    eps_train = eps_prev
                    high_eps_run = False

                # set the random training epsilon after each steps per collect
                # decay it by specified parameter every
                eps_train *= eps_decay
                eps_train = np.max([eps_train, eps_min])
                
                # log
                if result["n/ep"] > 0:
                    log_data = {"train":{
                            "episode": result["n/ep"],
                            "obs_reward": np.mean(result["rews"]),
                            "length": result["len"],
                            "exploration rate": eps_train,
                            "episodes": episodes_total
                        }
                    }
                    wandb.log(data=log_data, step=steps_total)
            
            # check test results
            set_eps(eps_test, single=True)
            policy.eval()
            
            passed_before = True
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
                    break
                else:
                    test_mazes.append(1)

            # log
            log_data = {"test":{
                    "obs_reward": np.mean(test_result["rews"]),
                    "length": test_result["len"],
                    "obs_reward_std": np.std(test_result["rews"]),
                }
            }
            wandb.log(data=log_data, step=steps_total)
            
            print(f"Evaluation Reward at Epoch {epoch+1}. Obs: {np.round(np.mean(test_result['rews']), 3)}, Maze: {seed}")
            
            print(f"Test Mazes results: {test_mazes}")

            # every n epochs render the policy for human-based evalution
            if (epoch % 10) == 0:
                for seed in range(1, mazes+1):
                    watch({"options":{"maze_type":maze_type, "n_mazes":seed, "random":False}})
                # reset back to training mode
                policy.train()

            # check if the agent can auccessfully solve the maze (within some threshold)
            if passed_mazes:
                print(f"Agents solved the current maze and all previous mazes. Solved all mazes on epoch {epoch+1}.")

                # watch the results if successful in passing twice
                for seed in range(1, mazes+1):
                    watch({"options":{"maze_type":maze_type, "n_mazes":seed, "random":False}})
                # reset back to training mode
                policy.train()
                break
            
            # to see if the agent can do it consequtively
            if passed_before:
                # pass the test maze at least twice
                passed_mazes = True
                print("Passed once")
            
            # reset back to training
            set_eps(eps_train, single=True)
            policy.train()
            
    print('Finished Training.')
    wandb.finish(0)