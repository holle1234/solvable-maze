from solver import DeadEndCanceling
from maze import RandomMaze

if __name__ == "__main__":
    maze = RandomMaze((150, 80)).create_maze()
    solver = DeadEndCanceling(maze)
    solver.solve()
