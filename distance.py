import numpy as np
# from maze import Tmaze
# from parameters import Parameters
# from simulation import run_simulation
import matplotlib.pyplot as plt
# from logger import Logger

from queue import *


class SquareGrid:
    def __init__(self, width: int, height: int, walls=[]):
        self.width = width
        self.height = height
        self.walls = walls
    
    def in_bounds(self, id: (int, int)) -> bool:
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id: (int, int)) -> bool:
        return id not in self.walls
    
    def neighbors(self, id: (int, int))-> list((int, int)):
        (x, y) = id
        neighbors = [(x+1, y), (x-1, y), (x, y-1), (x, y+1)] # E W N S
        if (x + y) % 2 == 0: neighbors.reverse() # S N W E
        results = filter(self.in_bounds, neighbors)
        results = filter(self.passable, results)
        return results

def reconstruct_path(came_from, start, goal):

    current = goal
    path = []
    if goal not in came_from: # no path was found
        return []
    while current != start:
        path.append(current)
        current = came_from[current]
    return path


def index_openfield (state:int) :
    x = -1 # row
    y = -1 # column

    if state < 0 :
        print("Erreur : index_openfield : state negatif")
        break

    #deux première colone
    elif state < 12 :
        x = state%6
        y = state//6

    elif state = 12 :
        x = 0
        y = 3
    
    elif state = 13 :
        x = 4
        y = 3
    
    elif state = 14 :
        x = 5
        y = 3

    if (x == -1) or (y == -1) :
        print("Erreur : index_openfield : resultat negatif")
        break

    return (x,y)



def breadth_first_search(mazetype: str="OpenField", start: int, goal: int):
    """
    Calculates the distance from a path between the given start to the given end in the given maze

    Arguments
    ----------
        mazetype -- Union[LinearTrack,OpenField] from maze.py : class with the maze and the agent
        start -- ( int X int ) : state of the maze
        end -- ( int X int ) : state of the maze
   
    Returns
    ----------   
        dist -- int : distance between the state e1 and the state e2
    """
    # representing the maze as a grid
    if mazetype == "OpenField" :
        m =  OpenField()
        start = index_openfield(start)
        goal = index_openfield(goal)
        
    elif mazetype == "LinearTrack" :
        m =  LinearTrack()

    else :
        print("Erreur : breadth_first_search : mazetype unknown")
        break

    height = m.height
    width = m.width
    walls = m.walls
    grid = (height, width, walls)


    # calculate the path
    frontier = Queue()
    frontier.put(start)
    came_from = {}
    came_from[start] = None
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in grid.neighbors(current):
            if next not in came_from :
                frontier.put(next)
                came_from[next] = current
    
    path = reconstruct_path(came_from, start, goal)

    if (len(path) <= 0) :
        print("Erreur : breadth_first_search : path empty")
        break

    return len(path)
