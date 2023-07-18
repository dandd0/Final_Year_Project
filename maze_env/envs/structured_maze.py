from maze_env.envs.mazelib import Maze
from maze_env.envs.solve.Collision import Collision

"""
A library file that contains the structured mazes
"""

def maze_grids(seed):
    """
    first type of 'shape' I want the model to learn: 

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
    1111101
    1111101
    1111101
    1111101
    1000001
    1111111
    a long reverse L shape
    x5 mazes

    1111111
    1011101
    1011101
    1011101
    1011101
    1000001
    1111111
    a long U shape
    x5 mazes
    """