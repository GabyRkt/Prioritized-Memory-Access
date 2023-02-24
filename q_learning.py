import numpy as np
from typing_extensions import ParamSpecArgs
from typing import Union
from maze import LinearTrack, OpenField
from parameters import Parameters
from mazemdp.toolbox import softmax, egreedy, egreedy_loc, sample_categorical


"""==============================================================================================================="""

def update_q_table(st,at,r,stp1, m:Union[LinearTrack,OpenField], params:Parameters) :
    """ Updates the Q-table (Q-learning)

        Arguments
        ----------
            st -- int : the current state
            at -- int : the taken action
            r -- float : the obtained reward  
            stp1 -- int : the next state
            m -- Union[LinearTrack,OpenField] from maze.py : class with the maze and the agent
            params -- Parameters from parameters.py : class with the settings of the current simulation 
        
        Returns
        ----------    
    """

    m.etr[st,:] = 0
    m.etr[st,at] = 1
    delta = r + params.gamma * np.max( m.Q[stp1] ) - m.Q[st,at]
    m.Q = m.Q + delta * m.etr
    m.etr = m.etr * params.lmbda * params.gamma

    return

"""==============================================================================================================="""