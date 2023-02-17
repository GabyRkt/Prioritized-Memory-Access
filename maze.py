from mazemdp.mdp import Mdp
from mazemdp.maze import Maze
import numpy as np

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

    #quick access to the agent (mdp)
    self.mdp = self.maze.mdp

    self.nb_states = self.mdp.nb_states
    self.action_space = self.mdp.action_space

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
    self.exp_LastStp1 = np.empty( (self.nb_states, self.action_space.n) ) 
    self.exp_LastR = np.empty( (self.nb_states, self.action_space.n) ) 
    self.etr = np.zeros( (self.nb_states , self.action_space.n) )
    self.left2right = True
    self.needMat = np.zeros( (self.nb_states , self.nb_states ) )



#===============================================================================================================#


class OpenField :


  def __init__(self) :
    """ create a OpenField maze with all necessary attributes """

    #create environement (maze) 
    self.maze = Maze(width= 9, height= 6, last_states= [41] , walls= [13,14,15,34,44,43,42])
    
    self.walls = self.maze.walls
    self.last_states = self.maze.last_states

    #quick access to the agent (mdp)
    self.mdp = self.maze.mdp

    self.nb_states = self.mdp.nb_states
    self.action_space = self.mdp.action_space

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
    return self.mdp.reset(uniform=True)




  def reInit(self) :
    """ wipe all of the agent's matrix (used for debugging) """
    self.Q = np.zeros( (self.nb_states , self.action_space.n) ) 
    self.T = np.zeros( (self.nb_states , self.nb_states ) )
    self.listExp = np.empty( (0,4) ) 
    self.exp_LastStp1 = np.empty( (self.nb_states, self.action_space.n) ) 
    self.exp_LastR = np.empty( (self.nb_states, self.action_space.n) ) 
    self.etr = np.zeros( (self.nb_states , self.action_space.n) )
    self.needMat = np.zeros( (self.nb_states , self.nb_states ) )





