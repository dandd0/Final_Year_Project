import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, Box
import pygame
import pygame.freetype

import numpy as np
import matplotlib.pyplot as plt

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, OrderEnforcingWrapper, AssertOutOfBoundsWrapper

from maze_env.envs.mazelib import Maze
from maze_env.envs.generate.BacktrackingGenerator import BacktrackingGenerator
from maze_env.envs.solve.Collision import Collision
from maze_env.envs.solve.ShortestPath import ShortestPath

"""
| Actions                   | Discrete                                      |    
| Agents                    | `agents= ['observer', 'explorer']`            |
| Agents                    | 2                                             |
| Action Shape              | (1)                                           |
| Action Values             | [0, 3]                                        |
| Observation Shape         | (size, size, 3) / (1)                         |
| Observation Values        | [0,1]                                         |

The basic actions will be mapped as such:
0: No movement
1: Up
2: Down
3: Right
4: Left

The observation space for the observer will be the map in 3 channels. 
The first channel will represent the walls present in the maze, with 1 indicating the walls and 0 indicating an empty space.
The second channel will represent the agent's current location, with 1 indicating the location and 0 otherwise.
The third channel will represent the exit of the maze, with 1 indicating the location and 0 otherwise.

The observation space for the explorer will be the action space of the observer.
It will be a single scalar where the value represents the corresponding action as defined above.
Due to some issues with Tianshou and non-equal action/observation spaces, the observation shape of the explorer will be the same as that of the observer, 
but with only the index [0,0,0] taking on the value of the observation, with the rest being 0.
This is incredibly inefficient but there are weird bugs with gym.spaces.Dict and supersuit padding, so this was the easier option.


The action values for the observer and explorer will be the 4 cardinal directions.
As for a growing catalog, that will be handled entirely within the agent class and not the envirnonment.
The environment will take in a single or an array (or some iterable) of actions and perform them consecutively.

The environment will end when either of the following conditions are met:
    - The agent reaches the exit
    - The steps exceed the episode max length
        - The max length of the episode will be defined as the total number of free spaces in the maze, which is: 2*(self.true_size**2) - 1
"""

def env(**kwargs):
    """
    In case I need to wrap the environment

    render_mode: The render mode for the environment. (None, human, rgb_array, ansi)
    size: The size of the maze (WxH) (Default = 10) 
    """
    env = MazeEnv(**kwargs)
    env = AssertOutOfBoundsWrapper(env) # make sure no invalid actions are given to the environment (0 to 3)
    env = OrderEnforcingWrapper(env) # make sure env.reset() is called before anything is done 
    # bad order enforcing shouldnt occur with the tianshou.env.PettingZooEnv wrapper but i'm adding it anyways for redundancy 

    return env

class MazeEnv(AECEnv):
    # define metadata for the environment, mainly render modes

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], 
                "render_fps":60, 
                "name":"maze_v0"}

    def __init__(self, render_mode=None, size=10):
        super().__init__()

        self.size = size # Raw W/H of the maze
        self.true_size = size*2 + 1 # the grid dimensions
        
        self.agents = ["observer", "explorer"] # names of the two agents
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        ) # mapping of the agent names to a number
    
        self._action_to_direction = {
            0: np.array([0,0]), # no movement
            1: np.array([-1,0]), # up
            2: np.array([1,0]), # down
            3: np.array([0,1]), # right
            4: np.array([0,-1]) # left
        }

        # check render mode is valid
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # define the space parameters for observation and action
        # only 5 possible primary actions for the 2 agents
        # observer can only tell (primary action-wise) to move up, down, right, left (no 'no movement' allowed)
        # explorer can only move up, down, right, left (no 'no movement' allowed)
        self.action_spaces = {agent: Discrete(5) for agent in self.agents}

        self.observation_spaces = {
            # observer & listener
            self.agents[0]: # the maze in 3 channels as described earlier.
                Box(low=0, high=1, shape=(3, self.true_size, self.true_size), dtype=np.int8),
            self.agents[1]:
                Box(low=0, high=3, shape=(1,1,1), dtype=np.int8)
        }

        # set the initial values for rewards, terminations, truncations and infos (all none or 0)
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"action_mask": Box(low=0, high=1, shape=(5,), dtype=bool)} for agent in self.agents}

        # set up the iterating agent selector
        self.agent_selection = None

        # define some self variables for future reference
        self.maze = None
        self.grid = None
        self.agent_loc = None
        self.exit = None
        self.maxlen = None
        self.minlen = None
        self.solution = None
        self.start = None

        # the action from the observer to the explorer
        self.message = None

        # precaution check purposes
        self._has_reset = False
        
        # rendering variables
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = 512

    # define observation space
    def observation_space(self, agent):
        "Get the observation space of the specified agent"
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        "Get the action space of the specified agent"
        return self.action_spaces[agent]
    
    def _legal_moves(self):
        legal_moves = []
        agent_loc_r, agent_loc_c = self.agent_loc

        # check if agent is at an edge
        if agent_loc_r == 0:
            # can only move down
            legal_moves = [2]
        elif agent_loc_r == self.grid.shape[0]-1:
            # can only move up
            legal_moves = [1]
        elif agent_loc_c == 0:
            # can only move right
            legal_moves = [3]
        elif agent_loc_c == self.grid.shape[1]-1:
            # can only move left
            legal_moves = [4]
        # everything else
        else:
            # check if can move up (or its exit)
            if (self.grid[agent_loc_r-1, agent_loc_c] == 0) or ((agent_loc_r-1, agent_loc_c) == self.exit):
                legal_moves += [1]
            # down
            if (self.grid[agent_loc_r+1, agent_loc_c] == 0) or ((agent_loc_r+1, agent_loc_c) == self.exit):
                legal_moves += [2]
            # right
            if (self.grid[agent_loc_r, agent_loc_c+1] == 0) or ((agent_loc_r, agent_loc_c+1) == self.exit):
                legal_moves += [3]
            # left
            if (self.grid[agent_loc_r, agent_loc_c-1] == 0) or ((agent_loc_r, agent_loc_c-1) == self.exit):
                legal_moves += [4]
        
        return legal_moves
    
    def _get_action_mask(self, agent):
         # because they technically 'share' the same action space (u,d,l,r), action masking is the same
        legal_moves = self._legal_moves() if agent == self.agent_selection else []
        action_mask = np.zeros(5, "int8")
        for i in legal_moves:
            action_mask[i] = 1
        
        return action_mask
    
    def observe(self, agent):
        "Get the current observation of the specified agent"

        if agent == self.agents[0]:
            # observer
            observation = np.zeros((3, self.grid.shape[0], self.grid.shape[1]))
            # set the grid observation
            observation[0,:,:] = self.grid
            # set agent location
            observation[1, self.agent_loc[0], self.agent_loc[1]] = 1
            # set exit location
            observation[2, self.exit[0], self.exit[1]] = 1
        elif agent == self.agents[1]:
            # explorer
            observation = np.array([[[self.message]]])
        else:
            observation = []

        action_mask = self._get_action_mask(agent)
        
        self.infos[agent]['action_mask'] = action_mask

        return observation
    
    def _generate_maze(self, seed=None):
        maze = Maze(seed) 
        maze.generator = BacktrackingGenerator(self.size, self.size) # depth first search maze
        maze.generate()
        maze.generate_entrances(False, False) # the entrances can be anywhere in the maze
        maze.solver = Collision() 
        maze.solve() # find the shortest path using collision (flood method)

        self.maze = maze # the maze object itself
        self.grid = maze.grid # the grid representation of the maze
        self.agent_loc = maze.start # the agent's starting location is the entrance
        self.exit = maze.end # the exit location
        self.maxlen = 2.5*(2*(self.size**2) - 1) # max exploration length before truncation (2.5* the total number of free grids in the maze)
        self.minlen = len(maze.solutions[0]) + 1 # the best possible path length for the agent to take ( +1 to reach the exit)
        self.solution = maze.solutions[0]
        self.start = maze.start
        # the above 2 has 0 index because there can be multiple solutions of equal minimum length, but we only need one so whatever

    def _generate_maze_trivial(self, seed=None):
        maze = Maze(seed) 
        maze.grid = np.ones((self.true_size, self.true_size))
        maze.grid[1:-1, 1:-1] = np.zeros((self.true_size-2, self.true_size-2))
        maze.generate_entrances() # the entrances are at the edges (opposite sides)
        
        # since this is trivial, there is no need for a solver (just use manhattan dist)

        self.maze = maze # the maze object itself
        self.grid = maze.grid # the grid representation of the maze
        self.agent_loc = maze.start # the agent's starting location is the entrance
        self.exit = maze.end # the exit location
        self.maxlen = 5*((self.true_size-2)**2) # max exploration length before truncation (2.5* the total number of free grids in the maze)
        self.minlen = np.sum(np.abs(np.array(self.agent_loc) - np.array(self.exit))) # manhattan dist for min length
        self.solution = None # no solution
        self.start = maze.start
        # the above 2 has 0 index because there can be multiple solutions of equal minimum length, but we only need one so whatever

    def select_maze(self, n_mazes, random=True):
        # n_mazes refers to how many of the mazes to randomly choose from
        # the seeds were chosen through generating and human selection based on difficulty.
        # the seeds are ordered from level of difficulty (based on subjective criteria, and within difficulty ordered by length)
        # each difficult has 8 mazes.
        maze_seeds = [2,23,10,24,26,13,17,30,22,4,18,3,8,38,48,0,32,11,1,25,35,27,28,29,33,43]

        if random:
            # randomly select on the given seeds
            maze_seed = np.random.choice(maze_seeds[:n_mazes])
        else:
            # else, select the specified maze
            maze_seed = maze_seeds[n_mazes]

        self._generate_maze(maze_seed)

    def reset(self, seed=None, options=None):
        # set up the maze
        if isinstance(options, dict):
            if options.get('maze_type'):
                if options['maze_type'] == "trivial":
                    self._generate_maze_trivial(seed)
                else:
                    self.select_maze(1)
            elif options.get('n_mazes'):
                assert options['n_mazes'] > 0, "NUMBER OF MAZES SET TO LESS THAN 1"
                self.select_maze(options['n_mazes'])
            elif options.get('maze_select'):
                # chooose specified maze
                self.select_maze(options['maze_select'], random=False)    
            else:
                self.select_maze(1)
        else:
            self.select_maze(1)

        self.prev_locs = [self.start]
        self.num_moves = 0
        self._has_reset = True
        
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}

        # select the first agent
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # define environment variables
        self.agents = self.possible_agents[:]
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"action_mask":self._get_action_mask(agent)} for agent in self.agents}

        self.message = 0 # no movement

        # reset seed
        np.random.seed()

    # make this modular in case i want to change the rewards function
    def _get_rewards(self):
        # check if agent is at exit
        if self.agent_loc == self.exit:
            # the reward will be the minimum number of steps needed to reach the end
            # (as determined by breadth first search)
            self.rewards = {agent: self.minlen for agent in self.agents}

        # else (i.e. not at exit), -1 to the rewards
        else:
            self.rewards = {agent: -1 for agent in self.agents}

    def _get_rewards2(self, action):
        # the explorer only needs to follow directions. the reward will be good if the agent has the same action as the observer
        # the observer has the same reward scheme as _get_rewards()S
        agents = self.agents

        # if the action of the explorer corresponds correctly to the observer (but not at exit)
        if self.message == action:
            # check if agent is at exit
            if self.agent_loc == self.exit:
                # the reward will be the minimum number of steps needed to reach the end
                # (as determined by breadth first search)
                self.rewards = {agents[0]: self.minlen,
                                agents[1]: 1
                                }
            else:
                self.rewards = {agents[0]: -1,
                                agents[1]: 1}
        
        else:
            # check if agent is at exit
            if self.agent_loc == self.exit:
                # the reward will be the minimum number of steps needed to reach the end
                # (as determined by breadth first search)
                self.rewards = {agents[0]: self.minlen,
                                agents[1]: -1
                                }
            else:
                self.rewards = {agents[0]: -1,
                                agents[1]: -1}
        
    def _get_rewards3(self, action):
        # the explorer only needs to follow directions. the reward will be good if the agent has the same action as the observer
        # the observer will be punished more severely if it returns to a previous explored square
        self.rewards = {agent: 0 for agent in self.agents}

        # observer
        if self.agent_loc == self.exit:
            self.rewards[self.agents[0]] = self.minlen # if reached exit
        elif self.agent_loc in self.prev_locs:
            self.rewards[self.agents[0]] = -2 # double penalty for repeated space
        else:
            self.rewards[self.agents[0]] = -1 # standard move

        # explorer
        # if action of observer corresponds to message, give 0, otherwise punish
        if self.message == action:
            self.rewards[self.agents[1]] = 0
        else:
            self.rewards[self.agents[1]] = -1

    def _get_rewards4(self, action):
        # same as rewards3, but with scaled rewards 
        # worst rewards will be -1
        # best reward will closer to 1
        self.rewards = {agent: 0 for agent in self.agents}

        # observer
        if self.agent_loc == self.exit:
            self.rewards[self.agents[0]] = 1 # if reached exit
        elif self.agent_loc in self.prev_locs:
            self.rewards[self.agents[0]] = -2/(self.maxlen*2) # double penalty for repeated space
        else:
            self.rewards[self.agents[0]] = -1/(self.maxlen*2) # standard move

        # explorer
        # if action of observer corresponds to message, give 0, otherwise punish
        if self.message == action:
            self.rewards[self.agents[1]] = 0
        else:
            self.rewards[self.agents[1]] = -1/(self.maxlen)
    
    def _check_termination(self):
        # when its at exit, end the game
        if self.agent_loc == self.exit:
            self.terminations = {agent: True for agent in self.agents}

    def _check_truncation(self):
        # when the number of moves surpass a maximum limit
        if self.num_moves > self.maxlen:
            self.truncations = {agent: True for agent in self.agents}

    # action is 0-3 indicating u, d, r, l respectively
    def step(self, action):
        # catch bad actions
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            return self._was_dead_step(action)

        agent = self.agent_selection

        legal_moves = self._get_action_mask(agent)
        assert legal_moves[action] == 1, "ILLEGAL MOVE"
        
        # when the agent is the observer
        if agent == self.agents[0]:
            # no rewards are allocated until the explorer moves
            self._clear_rewards()

            # set the message
            self.message = action
        
        # when the agent is the explorer
        elif agent == self.agents[1]:
            # NOTE THIS IS ONLY FOR SINGLE ACTIONS, IMPLEMENT ITERABLE ACTIONS LATER
            direction = self._action_to_direction[action]
            new_loc = self.agent_loc + direction
            new_loc_tup = tuple(new_loc)
            
            # one move (one move means one message. if its compound message, its still one move)
            self.num_moves += 1

            # set the rewards of both agents to be the same
            self.agent_loc = new_loc_tup
            self._get_rewards4(action)

            # check if agent has reached the exit or not
            self._check_termination()

            # check truncation (i.e. when its lost)
            self._check_truncation()

            # store previous locations
            self.prev_locs.append(new_loc_tup)

            # reset message
            self.message = 0 # no movement


        # switch selection to next agent
        self.agent_selection = self._agent_selector.next()

        # add rewards collected to cumulative rewards
        self._accumulate_rewards()

        # if human render mode, create a render
        if self.render_mode == "human":
            self.render()
    
    def _render_frame(self):
        if self.render_mode == "human" or "rgb_array":
            # initialize the game window
            if self.window is None and self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size+20)
                )
            
            if self.clock is None and self.render_mode == "human":
                self.clock = pygame.time.Clock()

            # draw the canvas
            font = pygame.freetype.SysFont(pygame.freetype.get_default_font(), 16)
            canvas = pygame.Surface((self.window_size, self.window_size+20))
            canvas.fill((255,255,255)) # make it white
            pix_square_size = (self.window_size/self.true_size) # size of each grid space

            # draw maze walls
            x,y = np.where(self.grid == 1)
            for x1, y1 in zip(x,y):
                pygame.draw.rect(
                    canvas,
                    (0,0,0),
                    pygame.Rect(
                    pix_square_size * y1,
                    pix_square_size * x1,
                    pix_square_size,
                    pix_square_size
                    )
                )

            # draw exit
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    (pix_square_size * self.exit[1], pix_square_size * self.exit[0]),
                    (pix_square_size, pix_square_size),
                ),
            )

            # draw start
            pygame.draw.rect(
                canvas,
                (0, 0, 255),
                pygame.Rect(
                    (pix_square_size * self.start[1], pix_square_size * self.start[0]),
                    (pix_square_size, pix_square_size),
                ),
            )

            # draw optimal solution
            # check if its trivial maze type
            if self.solution:
                for y, x in self.solution:
                    pygame.draw.circle(
                        canvas,
                        (163, 73, 164),
                        ((x + 0.5) * pix_square_size, (y + 0.5) * pix_square_size),
                        pix_square_size / 9,
                    )

            # draw agent
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                ((self.agent_loc[1] + 0.5) * pix_square_size, (self.agent_loc[0] + 0.5) * pix_square_size),
                pix_square_size / 3,
            )

            # draw message
            direction = self._action_to_direction[self.message]
            new_loc = self.agent_loc + direction
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                ((new_loc[1] + 0.5) * pix_square_size, (new_loc[0] + 0.5) * pix_square_size),
                pix_square_size / 6,
            )

            # num of moves so far
            font.render_to(canvas, (10, 514), "Number of Moves: " + str(self.num_moves), (0,0,0))

            # rewards so far
            font.render_to(canvas, (210, 514), "Obs Rew: " + str(np.round(self._cumulative_rewards['observer'], 3)), (0,0,0))
            font.render_to(canvas, (360, 514), "Exp Rew: " + str(np.round(self._cumulative_rewards['explorer'], 3)), (0,0,0))


            # update the game window
            if self.render_mode == "human":
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()

                # update clock
                self.clock.tick(self.metadata["render_fps"])
            elif self.render_mode == "rgb_array":
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                )
        else:
            pass

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return
        
        assert self._has_reset, "Reset the environment before calling .render()"

        if self.render_mode == "human" or "rgb_array":
            self._render_frame()
        elif self.render_mode == "ansi":
            print(self.grid)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
