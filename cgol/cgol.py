"""Conway's Game of Life

The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970. It is a zero-player game, meaning that its evolution is determined by its initial state, requiring no further input. One interacts with the Game of Life by creating an initial configuration and observing how it evolves. It is Turing complete and can simulate a universal constructor or any other Turing machine.

The new_world of the Game of Life is [a ...], two-dimensional orthogonal grid of [...] cells, each of which is in one of two possible states, live or dead [...]. Every cell interacts with its eight neighbours, which are the cells that are horizontally, vertically, or diagonally adjacent. At each step in time, the following transitions occur:

- Any live cell with fewer than two live neighbours dies, as if by underpopulation.
- Any live cell with two or three live neighbours lives on to the next generation.
- Any live cell with more than three live neighbours dies, as if by overpopulation.
- Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

[...]
The initial pattern constitutes the seed of the system. The first generation is created by applying the above rules simultaneously to every cell in the seed; births and deaths occur simultaneously, and the discrete moment at which this happens is sometimes called a tick. Each generation is a pure function of the preceding one. The rules continue to be applied repeatedly to create further generations.

- Wikipedia on Conway's game of life [adapted for p4p]
"""
import time

import numpy as np

def init_world(shape):
    """initialise a new CGoL world with random boolean entries

    Hint: Either use the np.empty function for not so random but rather
    arbitrary values or the more advanced np.random module. The
    np.empty function will suffice at the moment, though (have a
    look at the documentation online).

    Args:
        shape (tuple):  the number of rows and columns of the world

    Returns:
        (np.array): a CGoL world with random boolean entries.
    """
    # Write your method here and make sure the return type matches the one
    # given in the docstring.
    world = np.empty(shape, dtype=bool)
    world[:] = np.random.choice([True, False], size=shape)
    return world



def neighbours(world):
    """returns an array of identical shape as world with the number of alive
       cells adjacent to the cell.

    Args:
        world (np.array):  the cgol world

    Returns:
        (np.array): an array of identical shape as world with each 
                      element being the number of living neighbouring cells.
    Example:
        >>> world = init_world([3,3])
        >>> world.fill(False) # Make every entry False
        >>> world[2,1] = True
        >>> world[2,2] = True
        >>> world
        <<< array([[False, False, False],
        <<<        [False, False, False],
        <<<        [False,  True,  True]])
        >>>
        >>> neighbours(array)
        >>>
        <<< array([[0., 0., 0.],
        <<<        [1., 2., 2.],
        <<<        [1., 1., 1.]])
    """
    rows, cols = world.shape
    result = np.zeros_like(world, dtype=int)

    for i in range(rows):
        for j in range(cols):

            neighbors_count = np.sum(world[i - 1:i + 2, j - 1:j + 2]) - world[i, j]

            result[i, j] = neighbors_count

    return result


def survival(x, y, universe):
    """
    Compute one iteration of Life for one cell.

    :param x: x coordinate of cell in the universe
    :type x: int
    :param y: y coordinate of cell in the universe
    :type y: int
    :param universe: the universe of cells
    :type universe: np.ndarray
    """
    xf = x - 1
    if xf < 0: xf = 0
    xt = x + 2
    yf = y - 1 
    if yf < 0: yf = 0
    yt = y + 2
    num_neighbours = np.sum(universe[xf:xt, yf:yt]) - universe[x, y]
    # The rules of Life
    if universe[x, y] and not 2 <= num_neighbours <= 3:
        return 0
    elif num_neighbours == 3:
        return 1
    return universe[x, y]


def update(universe):
    """
    Compute one iteration of Life for the universe.

    :param universe: initial universe of cells
    :type universe: np.ndarray
    :return: updated universe of cells
    :rtype: np.ndarray
    """
    new_universe = np.copy(universe)
    # Apply the survival function to every cell in the universe
    for i in range(universe.shape[0]):
        for j in range(universe.shape[1]):
            new_universe[i, j] = survival(i, j, universe)
    return new_universe




def _run(world, cycles=1):
    """update the world <cycles> number of times

    Args:

        world (np.array):  the world to be updated
        cycles (int):      the number of cycles to go through

    Returns:
        new_world (np.array): the state of the world after <cycles> updates.
    """
    # Write your method here and make sure the return type matches the one
    # given in the docstring.
    new_world = np.copy(world)

    for _ in range(cycles):
        new_world = update(new_world)

    return new_world

def load(filename):
    """Load a saved pattern (Hint: Pay attention to the delimiter that separates values. Here, we want to use a comma (,))

    Args:
        filename (str): the path to a saved game (csv)

    Returns:
        (np.array): a CGoL world
    """
    loaded_data = np.genfromtxt(filename, delimiter=',', dtype=float)

    return loaded_data


def save(filename, world):
    """Save pattern to filename as csv (Hint: Pay attention to the delimiter that separates values. Here, we want to use a comma (,).

    Args:
        filename (str): the path to save the world to.
        world (np.array): the world to save
    """
    np.savetxt(filename, world, delimiter=',', fmt='%d')


def main():
    """the game

    Here, everything is assembled.
    A random world is initialised by passing a shape and using the init_world function or 
    an existing world is loaded by passing a filename,
    and runs at 2 iterations per second until a KeyboardInterrupt is risen.
    It then queries a filename to save to and leaves.
    """

    initialize_random = input("Do you want to initialize a random world? (y/n): ").lower() == 'y'
        
    if initialize_random:
        world = init_world((5, 5))  
    else:
        filename = input("Enter the filename to load the world from: ")
        world = load(filename)

    while True:
        print(world)
        world = update(world)

        time.sleep(0.5)



if __name__ == "__main__":
    main()
