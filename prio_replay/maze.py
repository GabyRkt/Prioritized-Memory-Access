from mazemdp.mdp import Mdp
from mazemdp.maze import Maze
from mazemdp.toolbox import N,S,E,W, sample_categorical
import numpy as np
import random

""" 
This file contains 2 classes : LinearTrack and OpenMaze, 
which utilises the mazemdp library to recreate the environments used in
Mattar & Daw's article on Prioritized Replay. 
"""


#===============================================================================================================#

class LinearTrack :

  def __init__(self) : 
    """ create a LinearTrack maze with all necessary attributes """

    #create environement (maze) 
    
    self.maze = Maze(width= 10, height=4, last_states= [1,18], walls= [1,2,5,6,9,10,13,14,17,18,21,22,25,26,29,30,33,34,37,38], hit=False)

    self.left2right = True

    self.walls = self.maze.walls
    self.last_states = self.maze.last_states
    self.nb_states = self.maze.nb_states
    self.action_space = self.maze.action_space

    #custom building the mdp , without a "well" state
    tmp_t = self.init_transitions(False)
    tmp_r = self.simple_reward(tmp_t)

    self.mdp = Mdp(
            self.nb_states,
            self.action_space,
            self.maze.mdp.P0,
            tmp_t,
            tmp_r,
            self.maze.mdp.plotter,
            gamma = self.maze.gamma,
            terminal_states=self.last_states,
            timeout=50,
        )

    #create State-Value Matrix (Q)
    self.Q = np.zeros( (self.nb_states , self.action_space.n) ) 

    #create Transition Matrix (T)
    self.T = np.zeros( (self.nb_states , self.nb_states ) )

    #create list of all experiences made by the agent
    self.listExp = np.empty( (0,4) ) 

    self.exp_LastStp1 = np.empty( (self.nb_states, self.action_space.n) ) 
    self.exp_LastR = np.empty( (self.nb_states, self.action_space.n) ) 

    #create eligibility trace matrix
    self.etr = np.zeros( (self.nb_states , self.action_space.n) )

    #create need matrix (state - state)
    self.needMat = np.zeros( (self.nb_states , self.nb_states ) ) 



  def reset(self) :
    """to be called at the end of an episode : reset the agent and returns it's starting state"""
    self.mdp.timestep = 0
    self.mdp.last_action_achieved = False

    if self.left2right :
      self.mdp.current_state = 0
    else :
      self.mdp.current_state = 19
    
    self.left2right = not self.left2right
    return self.mdp.current_state



  def reInit(self) :
    """ wipe all of the agent's matrix (used for debugging) """
    self.Q = np.zeros( (self.nb_states , self.action_space.n) ) 
    self.T = np.zeros( (self.nb_states , self.nb_states ) )
    self.listExp = np.empty( (0,4) ) 
    self.exp_LastStp1 = np.full( (self.nb_states, self.action_space.n) , np.NaN) 
    self.exp_LastR = np.zeros( (self.nb_states, self.action_space.n) ) 
    self.etr = np.zeros( (self.nb_states , self.action_space.n) )
    self.left2right = True
    self.needMat = np.zeros( (self.nb_states , self.nb_states ) )

  def init_transitions(self, hit):
        """
        Init the transition matrix
        a "well" state is added that only the terminal states can get into
        """

        transition_matrix = np.empty((self.nb_states, self.action_space.n, self.nb_states))

        transition_matrix[:, N, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, S, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, E, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, W, :] = np.zeros((self.nb_states, self.nb_states))

        for i in range(self.maze.width):
            for j in range(self.maze.height):
                state = self.maze.cells[i][j]
                if not state == -1:

                    # Transition Matrix when going north (no state change if highest cells or cells under a wall)
                    if j == 0 or self.maze.cells[i][j - 1] == -1:
                        transition_matrix[state][N][state] = 1.0
                    else:  # it goes up
                        transition_matrix[state][N][self.maze.cells[i][j - 1]] = 1.0

                    # Transition Matrix when going south (no state change if lowest cells or cells above a wall)
                    if j == self.maze.height - 1 or self.maze.cells[i][j + 1] == -1:
                        transition_matrix[state][S][state] = 1.0
                    else:  # it goes down
                        transition_matrix[state][S][self.maze.cells[i][j + 1]] = 1.0

                    # Transition Matrix when going east (no state change if left cells or on the left side of a wall)
                    if i == self.maze.width - 1 or self.maze.cells[i + 1][j] == -1:
                        transition_matrix[state][E][state] = 1.0
                    else:  # it goes left
                        transition_matrix[state][E][self.maze.cells[i + 1][j]] = 1.0

                    # Transition Matrix when going west (no state change if right cells or on the right side of a wall)
                    if i == 0 or self.maze.cells[i - 1][j] == -1:
                        transition_matrix[state][W][state] = 1.0
                    else:  # it goes right
                        transition_matrix[state][W][self.maze.cells[i - 1][j]] = 1.0


        return transition_matrix


  def simple_reward(self, transition_matrix: np.array):
          
          reward_matrix = np.zeros((self.nb_states, self.action_space.n))
          
          for i in range(2) :
            for state in range(self.nb_states) : 
              if state not in self.last_states :
                for action in range(0,4) :
                  if transition_matrix[state,action,self.last_states[i]] == 1 :
                    reward_matrix[state, action] = 1.0

          return reward_matrix

#===============================================================================================================#


class OpenField :


  def __init__(self) :
    """ create a OpenField maze with all necessary attributes """

    #create environement (maze) 
    self.maze = Maze(width= 9, height= 6, last_states= [41] , walls= [13,14,15,34,44,43,42])
    
    self.walls = self.maze.walls
    self.last_states = self.maze.last_states
    self.nb_states = self.maze.nb_states
    self.action_space = self.maze.action_space

    #custom building the mdp , without a "well" state
    tmp_t = self.init_transitions(False)
    tmp_r = self.simple_reward(tmp_t)

    self.mdp = Mdp(
            self.nb_states,
            self.action_space,
            self.maze.mdp.P0,
            tmp_t,
            tmp_r,
            self.maze.mdp.plotter,
            gamma = self.maze.gamma,
            terminal_states=self.last_states,
            timeout=50,
        )

    #create State-Value Matrix (Q)
    self.Q = np.zeros( (self.nb_states , self.action_space.n) ) 

    #create Transition Matrix (T)
    self.T = np.zeros( (self.nb_states , self.nb_states ) )

    #create list of all experiences made by the agent
    self.listExp = np.empty( (0,4) ) 

    self.exp_LastStp1 = np.full( (self.nb_states, self.action_space.n) , np.NaN) 
    self.exp_LastR = np.zeros( (self.nb_states, self.action_space.n) ) 

    #create eligibility trace matrix
    self.etr = np.zeros( (self.nb_states , self.action_space.n) )

    #create need matrix (state - state)
    self.needMat = np.zeros( (self.nb_states , self.nb_states ) )

  


  def reset(self) : 
    """to be called at the end of an episode : reset the agent and returns it's starting state"""

    #choose a state at random
    prob = np.ones(self.nb_states) / (self.nb_states)
    self.mdp.current_state = sample_categorical(prob)
    
    #but not the last state
    while self.mdp.current_state in self.last_states :
      prob = np.ones(self.nb_states) / (self.nb_states)
      self.mdp.current_state = sample_categorical(prob)

    # self.mdp.current_state = 2

    self.timestep = 0
    self.last_action_achieved = False
    return self.mdp.current_state


  def reInit(self) :
    """ wipe all of the agent's matrix (used for debugging) """
    self.Q = np.zeros( (self.nb_states , self.action_space.n) ) 
    self.T = np.zeros( (self.nb_states , self.nb_states ) )
    self.listExp = np.empty( (0,4) ) 
    self.exp_LastStp1 = np.empty( (self.nb_states, self.action_space.n) ) 
    self.exp_LastStp1 =  np.full( (self.nb_states, self.action_space.n) , np.NaN) 
    self.exp_LastR = np.zeros( (self.nb_states, self.action_space.n) ) 
    self.etr = np.zeros( (self.nb_states , self.action_space.n) )
    self.needMat = np.zeros( (self.nb_states , self.nb_states ) )

  
  def init_transitions(self, hit):
        """
        Init the transition matrix
        a "well" state is added that only the terminal states can get into
        """

        transition_matrix = np.empty((self.nb_states, self.action_space.n, self.nb_states))

        transition_matrix[:, N, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, S, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, E, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, W, :] = np.zeros((self.nb_states, self.nb_states))

        for i in range(self.maze.width):
            for j in range(self.maze.height):
                state = self.maze.cells[i][j]
                if not state == -1:

                    # Transition Matrix when going north (no state change if highest cells or cells under a wall)
                    if j == 0 or self.maze.cells[i][j - 1] == -1:
                        transition_matrix[state][N][state] = 1.0
                    else:  # it goes up
                        transition_matrix[state][N][self.maze.cells[i][j - 1]] = 1.0

                    # Transition Matrix when going south (no state change if lowest cells or cells above a wall)
                    if j == self.maze.height - 1 or self.maze.cells[i][j + 1] == -1:
                        transition_matrix[state][S][state] = 1.0
                    else:  # it goes down
                        transition_matrix[state][S][self.maze.cells[i][j + 1]] = 1.0

                    # Transition Matrix when going east (no state change if left cells or on the left side of a wall)
                    if i == self.maze.width - 1 or self.maze.cells[i + 1][j] == -1:
                        transition_matrix[state][E][state] = 1.0
                    else:  # it goes left
                        transition_matrix[state][E][self.maze.cells[i + 1][j]] = 1.0

                    # Transition Matrix when going west (no state change if right cells or on the right side of a wall)
                    if i == 0 or self.maze.cells[i - 1][j] == -1:
                        transition_matrix[state][W][state] = 1.0
                    else:  # it goes right
                        transition_matrix[state][W][self.maze.cells[i - 1][j]] = 1.0


        return transition_matrix


  def simple_reward(self, transition_matrix: np.array):
          
          reward_matrix = np.zeros((self.nb_states, self.action_space.n))
          
          for state in range(self.nb_states) : 
            if state not in self.last_states :
              for action in range(0,4) :
                if transition_matrix[state,action,self.last_states[0]] == 1 :
                  reward_matrix[state, action] = 1.0

          return reward_matrix
  


#===============================================================================================================#

class Tmaze :


  def __init__(self) :
    """ create a OpenField maze with all necessary attributes """

    #create environement (maze) 
    self.maze = Maze(width= 5, height= 2, last_states= [5] , walls= [1,3,7,9])
    
    self.walls = self.maze.walls
    self.last_states = self.maze.last_states
    self.nb_states = self.maze.nb_states
    self.action_space = self.maze.action_space

    #custom building the mdp , without a "well" state
    tmp_t = self.init_transitions(False)
    tmp_r = self.simple_reward(tmp_t)

    self.mdp = Mdp(
            self.nb_states,
            self.action_space,
            self.maze.mdp.P0,
            tmp_t,
            tmp_r,
            self.maze.mdp.plotter,
            gamma = self.maze.gamma,
            terminal_states=self.last_states,
            timeout=50,
        )

    #create State-Value Matrix (Q)
    self.Q = np.zeros( (self.nb_states , self.action_space.n) ) 

    #create Transition Matrix (T)
    self.T = np.zeros( (self.nb_states , self.nb_states ) )

    #create list of all experiences made by the agent
    self.listExp = np.empty( (0,4) ) 

    self.exp_LastStp1 = np.full( (self.nb_states, self.action_space.n) , np.NaN) 
    self.exp_LastR = np.zeros( (self.nb_states, self.action_space.n) ) 

    #create eligibility trace matrix
    self.etr = np.zeros( (self.nb_states , self.action_space.n) )

    #create need matrix (state - state)
    self.needMat = np.zeros( (self.nb_states , self.nb_states ) )

  


  def reset(self) : 
    """to be called at the end of an episode : reset the agent and returns it's starting state"""

    #choose a state at random
    self.mdp.current_state = 3


    self.timestep = 0
    self.last_action_achieved = False
    return self.mdp.current_state


  def reInit(self) :
    """ wipe all of the agent's matrix (used for debugging) """
    self.Q = np.zeros( (self.nb_states , self.action_space.n) ) 
    self.T = np.zeros( (self.nb_states , self.nb_states ) )
    self.listExp = np.empty( (0,4) ) 
    self.exp_LastStp1 = np.empty( (self.nb_states, self.action_space.n) ) 
    self.exp_LastStp1 =  np.full( (self.nb_states, self.action_space.n) , np.NaN) 
    self.exp_LastR = np.zeros( (self.nb_states, self.action_space.n) ) 
    self.etr = np.zeros( (self.nb_states , self.action_space.n) )
    self.needMat = np.zeros( (self.nb_states , self.nb_states ) )

  
  def init_transitions(self, hit):
        """
        Init the transition matrix
        a "well" state is added that only the terminal states can get into
        """

        transition_matrix = np.empty((self.nb_states, self.action_space.n, self.nb_states))

        transition_matrix[:, N, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, S, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, E, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, W, :] = np.zeros((self.nb_states, self.nb_states))

        for i in range(self.maze.width):
            for j in range(self.maze.height):
                state = self.maze.cells[i][j]
                if not state == -1:

                    # Transition Matrix when going north (no state change if highest cells or cells under a wall)
                    if j == 0 or self.maze.cells[i][j - 1] == -1:
                        transition_matrix[state][N][state] = 1.0
                    else:  # it goes up
                        transition_matrix[state][N][self.maze.cells[i][j - 1]] = 1.0

                    # Transition Matrix when going south (no state change if lowest cells or cells above a wall)
                    if j == self.maze.height - 1 or self.maze.cells[i][j + 1] == -1:
                        transition_matrix[state][S][state] = 1.0
                    else:  # it goes down
                        transition_matrix[state][S][self.maze.cells[i][j + 1]] = 1.0

                    # Transition Matrix when going east (no state change if left cells or on the left side of a wall)
                    if i == self.maze.width - 1 or self.maze.cells[i + 1][j] == -1:
                        transition_matrix[state][E][state] = 1.0
                    else:  # it goes left
                        transition_matrix[state][E][self.maze.cells[i + 1][j]] = 1.0

                    # Transition Matrix when going west (no state change if right cells or on the right side of a wall)
                    if i == 0 or self.maze.cells[i - 1][j] == -1:
                        transition_matrix[state][W][state] = 1.0
                    else:  # it goes right
                        transition_matrix[state][W][self.maze.cells[i - 1][j]] = 1.0


        return transition_matrix


  def simple_reward(self, transition_matrix: np.array):
          
          reward_matrix = np.zeros((self.nb_states, self.action_space.n))
          
          for state in range(self.nb_states) : 
            if state not in self.last_states :
              for action in range(0,4) :
                if transition_matrix[state,action,self.last_states[0]] == 1 :
                  reward_matrix[state, action] = 1.0

          return reward_matrix