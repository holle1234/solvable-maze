import cv2
import numpy as np
from random import choice, shuffle, sample
import time
from dataclasses import dataclass
from typing import Iterable, Tuple, Set
from itertools import product



# for creating a randomized fully connected maze that is always solvable
class RandomMaze:
    def __init__(self, shape):
        self.shape = self.validate_shape(shape)
        self.maze = np.zeros(self.shape, np.uint8)

    def validate_shape(self, shape):
        # y must be even and x must be odd for maze to centered
        new_shape = list(shape)
        if shape[0] % 2 != 0:
            new_shape[0] = shape[0] + 1
        if shape[1] % 2 == 0:
            new_shape[1] = shape[1] + 1
        return np.array(new_shape)

    def create_maze(self):
        # dict of possible steps and neighbouring steps
        valid_steps = self.get_valid_steps()
        # select a random drawing point
        start_point = choice(list(valid_steps.keys()))
        # holds steps taken that could have free neighbours
        intersections = [start_point]
        # holds all steps taken, quick check if point has been visited
        visited = {start_point}

        # iterate intersections using FIFO method
        # finds neighbours to a point that have not been selected yet
        # picks one of the free neighbours and sets it to be searched on the next iteration
        # produces a fully connected maze

        while intersections:
            point = intersections[-1]
            values = valid_steps[point] - visited
            if values:
                new_point = values.pop()
                visited.add(new_point)
                cv2.line(self.maze, point, new_point, 1, 1)
                intersections.append(new_point)
                if not values:
                    intersections.pop(-2)
            else:
                intersections.pop()
        self.create_entrance_exit()
        return self.maze

    def get_valid_steps(self):
        # valid steps with valid neighbours
        h, w = self.shape
        points = {}
        rh, rw = range(0, h, 2), range(1, w - 1, 2)
        for hh, ww in product(rh, rw):
            points[(hh, ww)] = []
        for hh, ww in points.keys():
            n = [(hh + 2, ww), (hh - 2, ww), (hh, ww + 2), (hh, ww - 2)]
            n = {i for i in n if i in points}
            points[(hh, ww)] = n
        return points

    def create_entrance_exit(self):
        # make entrance
        first_row = self.maze[2] > 0
        inds = np.arange(len(first_row))[first_row]
        self.maze[:2,  choice(inds)] = 1

        # make exit
        last_row = self.maze[-2] > 0
        inds = np.arange(len(last_row))[last_row]
        zeros = np.zeros_like(self.maze[-1])
        zeros[choice(inds)] = 1
        self.maze = np.vstack((self.maze, zeros))





if __name__ == "__main__":
    t = time.time()
    maze = RandomMaze((500, 500))
    print(time.time() - t)
    solver = DeadEndCanceling(maze)
    solver.solve()