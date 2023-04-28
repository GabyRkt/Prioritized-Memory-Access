import numpy as np
from typing_extensions import ParamSpecArgs
from typing import Union

from prio_replay.maze import LinearTrack, OpenField
from prio_replay.parameters import Parameters
from mazemdp.toolbox import sample_categorical
from prio_replay.q_learning import get_action


"""==============================================================================================================="""


def create_plan_exp( m : Union[LinearTrack,OpenField], params : Parameters ) :
    """ Creates a matrix of tuple (state, action, reward, next_state) based on the previous experiences

        Arguments
        ----------
            m -- Union[LinearTrack,OpenField] from maze.py : class with the maze and the agent
            params -- Parameters from parameters.py : class with the settings of the current simulation 
        
        Returns
        ----------    
            planExp -- matrix ((state X action) X 4) : memory of last reward and next state obtained of each tuple (state, action)
    """


    """
    create a matrix with shape :
     [
       [ [ s[0] a[0] last_r[0,0] last_stp1[0,0] ] ]
       [ [ s[1] a[0] last_r[1,0] last_stp1[1,0] ] ]
       [ [ s[2] a[0] last_r[2,0] last_stp1[2,0] ] ]
                     ....
       [ [ s[0] a[1] last_r[0,1] last_stp1[0,1] ] ]
       [ [ s[1] a[1] last_r[1,1] last_stp1[1,1] ] ]
       [ [ s[2] a[1] last_r[2,1] last_stp1[2,1] ] ]
                  ..........
     ]
    """
    planExp = np.empty( (0,4) )

    for a in range(m.action_space.n) :
        for s in range(m.nb_states) :
            exp = np.array( [ s, a, m.exp_LastR[s,a], m.exp_LastStp1[s,a] ] ) 
            planExp = np.append( planExp, [exp], axis=0 )
    
    if params.remove_samestate : # remove all experiences that goes to the same states (i.e walls)
        planExp = planExp[planExp[:, 0] != planExp[:, 3]]
    
    planExp = planExp[ np.invert( np.isnan(planExp).any(axis=1) ) ] # remove all experiences with NaNs in it , we dont need this theoretically

    return planExp

"""==============================================================================================================="""

def expand_plan_exp(planning_backups, planExp, m : Union[OpenField,LinearTrack], params) :

    seqStart = np.argwhere(planning_backups[:, 4] == 1)[-1][0]
    seqSoFar = planning_backups[seqStart:, 0:4]
    
    sn = int( seqSoFar[-1, 3] )  # Final state reached in the last planning st...

    if params.onlineVSoffline == "online" : # agent is awake 
        an = get_action(sn,m,params)

    else : # agent is asleep 
        probs = np.zeros( np.size(m.Q[sn]) )
        probs[ m.Q[sn] == max( m.Q[sn] ) ] =  1 / ( sum(m.Q[sn]) == max(m.Q[sn]) ) 
        an = sample_categorical(probs)

    snp1 = m.exp_LastStp1[sn,an]
    rn = m.exp_LastR[sn,an]

    step_isNaN = np.isnan( m.exp_LastStp1[sn,an] ) or np.isnan( m.exp_LastR[sn,an] )
    step_isRepeated = np.isin( snp1 , [ seqSoFar[:, 0], seqSoFar[:, 3] ] )

    if (not step_isNaN) and (params.allowLoops or (not step_isRepeated)) :
        expanded_exp = np.array( [sn, an, rn, snp1] )
        seqUpdated = np.append( seqSoFar, [expanded_exp], axis=0 )
        planExp.append(seqUpdated)


"""==============================================================================================================="""

def update_planning_backups(planning_backups, plan_exp_arr_max) :

    if planning_backups.shape[0] > 0:
        planning_backups = np.vstack([planning_backups, np.append(plan_exp_arr_max, plan_exp_arr_max.shape[0])])
    elif planning_backups.shape[0] == 0:
        planning_backups = np.append(plan_exp_arr_max,plan_exp_arr_max.shape[0]).reshape(1, planning_backups.shape[1])
    else:
        err_msg = 'planning_backups does not have the correct shape. It is {} but should have a length equal to 1 or 2, e.g. (5,) or (2, 5)'.format(planning_backups.shape)
        raise ValueError(err_msg)
    
    return planning_backups

"""==============================================================================================================="""