import numpy as np
from typing_extensions import ParamSpecArgs
from typing import Union

from prio_replay.maze import LinearTrack, OpenField
from prio_replay.parameters import Parameters
from mazemdp.toolbox import softmax, egreedy, egreedy_loc, sample_categorical


"""==============================================================================================================="""

def run_pre_explore(m : Union[LinearTrack,OpenField]) :
    """ Updates the transition matrix m.T for the first time

        Arguments
        ----------
            m -- Union[LinearTrack,OpenField] from maze.py : class with the maze and the agent
        
        Returns
        ----------      
    """
   
    # letting the agent try every action at each state, to build a transition matrix
    for s in range(0, m.nb_states) :
        for a in range(0,4) :
            if (s not in m.last_states) : # don't explore last state and well
                m.mdp.current_state = s
                [stp1, _, _, _] = m.mdp.step(a) 
                
                m.listExp = np.append( m.listExp , [[s, a, 0, stp1]], axis=0) # update the list of experiences  
                
                m.exp_LastStp1[s,a] = stp1 # update the list of last stp1 obtained with (st,at)
                m.exp_LastR[s,a] = 0 # update the list of last reward obtained with (st,at)

                m.T[s,stp1] = m.T[s,stp1] + 1
    
    # normalising the transition matrix
    for i_row in range(m.T.shape[0]) :
        m.T[i_row] = [float(i)/sum(m.T[i_row]) for i in m.T[i_row]]
    
    # dividing when sum(m.T[i_row])=0 causes NaN, so we replace NaNs here with 0
    m.T[ np.isnan(m.T) ] = 0

    return

"""==============================================================================================================="""

def add_goal2start(m: Union[LinearTrack,OpenField], params : Parameters) :
    """ Adds transitions between the goal and the potential starts to restart an episode

        Arguments
        ----------
            m -- Union[LinearTrack,OpenField] from maze.py : class with the maze and the agent
            params -- Parameters from parameters.py : class with the settings of the current simulation 
        
        Returns
        ----------     
    """

    for last_state in m.last_states :
        
        if params.start_rand :
            m.T[last_state,:] = 0 # at first, put all transitions from last_state to 0

            # get a list of all index of valid start states : i.e not last_states 
            l_valid_states = [ i for i in range(m.nb_states) if (i not in m.last_states)   ]

            # transitions from goal to all possible start states have the same probability
            for valid_state in l_valid_states :
                 m.T[last_state,valid_state] = 1/len(l_valid_states)
        
        else :
            m.T[last_state,:] = 0  # at first, put all transitions from last_state to 0
            
            # Top-Track Last State ==> Bottom-Track Start State
            if last_state == 18 :
                m.T[last_state,19] = 1
            # Bottom-Track Last State ==> Top-Track Start State
            elif last_state == 1 :
                m.T[last_state,0] = 1

            # Open Field
            elif last_state == 41 :
                m.T[last_state,2] = 1

            # T maze
            elif last_state == 5 :
                m.T[last_state,3] = 1

    return

"""==============================================================================================================="""

def update_transition_n_experience(st,at,r,stp1, m:Union[LinearTrack,OpenField], params:Parameters) :
    """ Udaptes the transition matrix and list of experience

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

    # Update transition matrix
    targVec = np.zeros( (1, m.nb_states) )
    targVec[0][stp1] = 1
    m.T[st,:] = m.T[st,:] + params.Talpha * ( targVec - m.T[st,:] ) # shift T-matrix towards targvec (?) => needs explanation

    # Update list of experience
    m.listExp = np.append( m.listExp , [[st, at, r, stp1]], axis=0) # update the list of experiences  
    m.exp_LastStp1[st,at] = stp1 # update the list of last stp1 obtained with (st,at)
    m.exp_LastR[st,at] = r # update the list of last reward obtained with (st,at)

    return

"""==============================================================================================================="""