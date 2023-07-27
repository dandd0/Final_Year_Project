from maze_env.envs.mazelib import Maze
from maze_env.envs.solve.Collision import Collision

import numpy as np

"""
A library file that contains the structured mazes
An absolutely horrible way of doing it, but idgaf anymore lmao

the shapes I want the model to learn: 

1111111
1011111
1011111
1011111
1011111
1000001
1111111
a long L shape
x5 mazes

1111111
1011111
1011111
1011111
1011111
1000001
1111111
a long reverse L shape (backwards)
x5 mazes

1111111
1110001
1110111
1000111
1011111
1011111
1111111
a zigzag shape
x5 mazes
"""

def structured_maze_grids(seed):
    """
    generate the structured mazes
    """
    maze = Maze()
    # L shape
    if seed == 1:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                                [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                                [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                                [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start = (7,3)
        maze.end = (11,7)
    elif seed == 2:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start = (11,3)
        maze.end = (11,9)
    elif seed == 3:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                                [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                                [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                                [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                                [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start = (1,5)
        maze.end = (7,11)
    elif seed == 4:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                                [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                                [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start = (1,5)
        maze.end = (9,5)
    elif seed == 5:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start = (3,3)
        maze.end = (11,11)
    # reverse L shape
    elif seed == 6:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start = (11,5)
        maze.end = (7,1)
    elif seed == 7:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                        [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)

        maze.start=(7,9)
        maze.end=(1,1)
    elif seed == 8:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start=(7,9)
        maze.end=(1,5)
    elif seed == 9:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                        [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start=(7,9)
        maze.end=(1,1)
    elif seed == 10:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start=(11,11)
        maze.end=(7,1)
    # zigzag
    elif seed == 11:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
                        [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start=(7,5)
        maze.end=(3,9)
    elif seed == 12:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                        [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start=(7,5)
        maze.end=(1,5)
    elif seed == 13:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                        [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start=(11,5)
        maze.end=(7,11)
    elif seed == 14:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                        [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                        [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start=(11,3)
        maze.end=(5,9)
    elif seed == 15:
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start=(11,3)
        maze.end=(3,11)
    else:
        print("Invalid maze seed, pls check")
        maze.grid = np.array([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)
        maze.start=(11,3)
        maze.end=(3,11)
            
    maze.solver = Collision()
    maze.solve()
    return maze