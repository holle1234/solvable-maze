import cv2
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple



# Treelike structure for retracing the steps of a solving path
@dataclass
class Node:
      value: Tuple[int, int]
      parent: None

class Tree:
    def __init__(self, init_values:Iterable[Tuple[int, int]]=None):
        self.heads = []
        if init_values:
            for i in init_values:
                self.heads.append(Node(value=i, parent=None))

    def add_head(self, value, parent):
        node = Node(value, parent=parent)
        self.heads.append(node)
        return node

    def gather(self, node):
        yield node.value
        while node.parent:
            node = node.parent
            yield node.value

    def remove(self, node):
        self.heads.remove(node)


# Path solver
class MazeRunner:
    def __init__(self, start_point, maze):
        self.maze = maze
        self.shape = maze.shape
        self.start_point = tuple(start_point)
        self.tree = Tree(init_values=[self.start_point])

    def get_steps(self):
        new_nodes = []
        for head in self.tree.heads.copy():
            v_steps = self.get_valid_steps(head.value)
            for v_step in v_steps:
                new_nodes.append(self.tree.add_head(v_step, parent=head))
            if not v_steps:
                self.tree.remove(head)
        return new_nodes

    def get_valid_steps(self, loc):
        h, w = self.shape
        y, x = loc
        steps = [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
        steps = [i for i in steps if (0 < i[0] < h) and (0 < i[1] < w)]
        steps = [i for i in steps if self.maze[i] == 1]
        return steps


# Solving strategy that handles user interface
class DeadEndCanceling:
    def __init__(self, maze:np.ndarray):
        self.maze = maze
        self.shape = np.array(self.maze.shape)
        self.window_size = tuple((self.shape * (750 / max(self.shape))).astype(np.int))[::-1]
        self.solvers = self.init_solvers()

    def init_solvers(self):
        solvers = []
        for start_point in self.get_start_indices():
            solvers.append(MazeRunner(start_point, self.maze))
            self.maze[start_point] = 2
        print("{} solvers initialized".format(len(solvers)))
        return solvers

    def get_start_indices(self):
        # find indices of unconnected start points on the top row
        row = self.maze[0]
        inds = np.arange(len(row))
        inds = inds[(row[(inds - 1)] == 0) & (row != 0)]
        zeros = np.zeros_like(inds, dtype=inds.dtype)
        return zip(zeros, inds)

    def solve(self):
        solution = None
        while self.solvers and not solution:
            for solver in self.solvers.copy():
                nodes = solver.get_steps()
                if not nodes:
                    self.solvers.remove(solver)

                for node in nodes:
                    step = node.value
                    self.maze[step] = 2
                    if step[0] >= self.shape[0] - 1:
                        solution = solver.tree.gather(node=node)
                        self.draw_solution(solution)
                        break

                img = self.get_rgb_img()
                cv2.imshow("img", img)
                cv2.waitKey(int(not bool(solution)))

    def draw_solution(self, solution):
        print("winner route found!")
        self.maze = np.where(self.maze == 2, 1, self.maze)
        steps = np.array([i for i in solution])
        self.maze[tuple(steps.T)] = 3

    def get_rgb_img(self):
        img = cv2.merge((self.maze, self.maze, self.maze))
        img[self.maze == 0] = [0, 0, 0] # color of the maze walls
        img[self.maze == 1] = [132,132,132] # color of the non-travelled path
        img[self.maze == 2] = [240, 240, 0] # color of the generic unfinished route
        img[self.maze == 3] = [0, 255, 255] # color of the winner route
        img = cv2.resize(img, self.window_size, interpolation=cv2.INTER_AREA)
        return img
