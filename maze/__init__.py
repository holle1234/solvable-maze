import cv2
import numpy as np
from random import choice, shuffle
from collections import OrderedDict
from itertools import zip_longest


# for creating a randomized fully connected maze that is always solvable
class RandomMaze:
    def __init__(self, shape):
        self.shape = shape
        self.maze = np.zeros(self.shape, np.uint8)

    def create_maze(self):
        # dict of possible steps and neighbouring steps
        converted_coords, data = self.get_valid_steps()
        # select a random drawing point
        start_point = choice(list(converted_coords.keys()))
        # holds steps taken that could have free neighbours
        intersections = [start_point]
        # holds all steps taken, quick check if point has been visited
        visited = {start_point}

        # Iterate intersections using FIFO method
        # Finds free neighbours to a point that have not been visited yet
        # Picks one of the free neighbours and sets it to be searched for the next iteration
        # Produces a fully connected maze

        while intersections:
            point = intersections[-1]
            values = converted_coords[point] - visited
            if values:
                new_point = values.pop()
                visited.add(new_point)
                cv2.line(self.maze, data[point], data[new_point], 1, 1)
                intersections.append(new_point)
                if not values:
                    intersections.pop(-2)
            else:
                intersections.pop()

        self.pad_right()
        self.pad_down()
        self.create_entrance_exit()
        return self.maze

    def get_valid_steps(self):

        # Sets cant be shuffled nor random samples cant be drawn
        # therefor custom data structure must be implemented.
        # 1. Coordinate-tuples are generated using randomized cartesian product.
        # 2. For set.pop method to work, each coordinate is given an index to hide the real value.
        # This method will avoid otherwise necessary type conversions between list, array and set.
        # Also avoids multiple random.choice calls which are more expensive than set.pop.

        h, w = self.shape
        hr, wr = range(1, h - 1, 2), range(1, w - 1, 2)
        points, con_table = OrderedDict(), OrderedDict()
        for ind, (hh, ww) in enumerate(self.randomized_product(wr, hr)):
            con_table[(hh, ww)] = ind
        for hh, ww in con_table.keys():
            n = [(hh + 2, ww), (hh - 2, ww), (hh, ww + 2), (hh, ww - 2)]
            points[con_table[(hh, ww)]] = {con_table[i] for i in n if i in con_table}
        return points, list(con_table.keys())

    def randomized_product(self, h, w):
        gens = [zip_longest([], w, fillvalue=i) for i in h]
        while gens:
            removables = set()
            for gen in gens:
                try:
                    yield next(gen)
                except StopIteration:
                    removables.add(gen)
            gens = list(set(gens) - removables)
            shuffle(gens)

    def create_entrance_exit(self):
        # make entrance
        _range = np.arange(1, self.shape[1] - 1, 2)
        self.maze[0,  choice(_range)] = 1
        # make exit
        self.maze[-1:, choice(_range)] = 1

    def pad_down(self):
        # beautify final rows of the maze if height is not even
        if self.shape[0] % 2 == 0:
            _range = np.arange(1, self.shape[1] - 1, 2)
            last_row = self.maze[-3].copy()
            sec_last_row = self.maze[-4].copy()
            self.maze[-3] = sec_last_row
            self.maze[-2] = last_row

            row = self.maze[-5]
            corners_upper = _range[(row[_range - 1] == 0) | (row[_range + 1] == 0)]
            corners_lower = np.setdiff1d(_range, corners_upper)

            self.maze[-4, corners_upper] = 1
            self.maze[-4, corners_lower[::2]] = 1
            self.maze[-3, corners_lower[1::2]] = 1

    def pad_right(self):
        if self.shape[1] % 2 == 0:
            h_range = np.arange(1, self.shape[0] - 1, 2)
            last_col = self.maze[:,-3].copy()
            sec_last_col = self.maze[:,-4].copy()
            self.maze[:,-3] = sec_last_col
            self.maze[:, -2] = last_col
            self.maze[h_range[::2], -4] = 1
            self.maze[h_range[1::2], -3] = 1



