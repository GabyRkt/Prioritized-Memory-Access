from maze import Tmaze, OpenField, LinearTrack
from parameters import Parameters

import matplotlib.pyplot as plt
from queue import *
import numpy as np


class SquareGrid:
    def __init__(self, width: int, height: int, walls=[]):
        self.width = width
        self.height = height
        self.walls = walls
    
    def in_bounds(self, id: tuple[int, int]) -> bool:
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id: tuple[int, int]) -> bool:
        return id not in self.walls
    
    def neighbors(self, id: tuple[int, int])-> list((int, int)):
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


def index_openfield (state:int , walls : bool = False) :
    x = -1 # row
    y = -1 # column

    if not walls :

        if state < 0 :
            print("Erreur : index_openfield : state negatif")
            return

        elif state < 13 :
            x = state%6
            y = state//6
        
        elif 12 < state < 31 :
            x = (state + 3 )%6
            y = (state + 3 )//6
        
        elif 30 < state < 38 :
            x = (state + 4 )%6
            y = (state + 4 )//6
        
        elif 37 < state < 47 :
            x = (state + 7 )%6
            y = (state + 7 )//6

        if (x == -1) or (y == -1) :
            print("Erreur : index_openfield : resultat negatif")
            return

    else : 

        if 0 <= state < 47 :
            x = state%6
            y = state//6

    return (x,y)


def breadth_first_search(start:int, goal:int, mazetype: str="OpenField"):
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
        start = index_openfield(start, walls=False)
        goal = index_openfield(goal, walls=False)
        
    elif mazetype == "LinearTrack" :
        m =  LinearTrack()

    else :
        print("Erreur : breadth_first_search : mazetype unknown")
        return

    height = m.maze.height
    width = m.maze.width
    
    walls = []
    for elem in m.walls :
        walls.append( index_openfield(elem, walls=True) )
    
    grid = SquareGrid(height, width, walls)


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

    if (len(path) < 0) :
        print("Erreur : breadth_first_search : path negative")
        return

    return len(path)

