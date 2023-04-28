from typing_extensions import ParamSpecArgs
from typing import Union
import numpy as np

from mazemdp.toolbox import softmax, egreedy, egreedy_loc, sample_categorical
from prio_replay.maze import LinearTrack, OpenField
from prio_replay.parameters import Parameters


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

def update_q_wplan (ep_i, st, p, log, step_i, prev_s, prev_stp1, plan_exp_arr_max, m, params):
    """ Updates Q-values using planExp with the highest EVB (plan_exp_arr_max)

        Arguments
        ----------
            st : current state of the agent
            p -- (int) : current number of planning step
            log -- (Object) : logger to store data 
            step_i -- (int) : current number of steps 
            plan_exp_arr_max -- (array) : array of experience with maximum EVB 
            m -- Union[LinearTrack,OpenField] from maze.py : class with the maze and the agent
            params -- Parameters from parameters.py : class with the settings of the current simulation 
        
        Returns
        ----------
    
    """
    
    for n in range(plan_exp_arr_max.shape[0]):

        # Retrieve information from this experience
        s_plan = int(plan_exp_arr_max[n][0])
        a_plan = int(plan_exp_arr_max[n][1])

        # Individual rewards from this step to end of trajectory
        rew_to_end = plan_exp_arr_max[n:][:, 2]
        # Notice the use of '-1' instead of 'n', meaning that stp1_plan is the final state of the
        # trajectory
        
        stp1_plan = int(plan_exp_arr_max[-1][3])

        # Discounted cumulative reward from this step to end of trajectory
        n_plan = np.size(rew_to_end)
        r_plan = np.dot(np.power(params.gamma, np.arange(0, n_plan)), rew_to_end)

        # Add plan and q_learning updates to q_learning function
        stp1_value = np.max(m.Q[stp1_plan])
        Q_target = r_plan + (params.gamma ** n_plan) * stp1_value
    
        log.nb_backups_per_state[ep_i][s_plan] += 1
        log.nb_replay_per_ep[ep_i] += 1

        # Update Q-value for this state-action pair 
        m.Q[s_plan, a_plan] = m.Q[s_plan, a_plan] + params.alpha * (Q_target - m.Q[s_plan, a_plan])

        # saving data about forward and reverse replay to plot figures
        if (p > 1) :
            if (s_plan != prev_s) and (st % 2) == (s_plan % 2):
                
                if step_i == 0 :
            
                    if s_plan == prev_stp1  :
                        log.nbStep_forwardReplay_forward += 1
                        log.forward[ep_i] += 1
                        
                        # log.nb_backups_per_state[ep_i][s_plan] += 1
                    
                    if stp1_plan == prev_s :
                        log.nbStep_backwardReplay_forward += 1
                        log.backward[ep_i] += 1
                        
                        # log.nb_backups_per_state[ep_i][s_plan] += 1

                else :
                    
                    if s_plan == prev_stp1  :
                        log.nbStep_forwardReplay_backward += 1
                        log.forward[ep_i] += 1
                        
                        # log.nb_backups_per_state[ep_i][s_plan] += 1
                    
                    if stp1_plan == prev_s :
                        log.nbStep_backwardReplay_backward += 1
                        log.backward[ep_i] += 1
                        
                        # log.nb_backups_per_state[ep_i][s_plan] += 1

    return s_plan, stp1_plan


"""==============================================================================================================="""

def get_action(st, m: Union[LinearTrack,OpenField], params : Parameters) :
    """ Determines which action to take for the current state based on the policy in the settings

        Arguments
        ----------
            st -- int : the current state
            m -- Union[LinearTrack,OpenField] from maze.py : class with the maze and the agent
            params -- Parameters from parameters.py : class with the settings of the current simulation 
        
        Returns
        ----------  
            at -- int : the action chosen based on the policy  
    """

    if params.actpolicy == "softmax" :
        probs = softmax(m.Q, st, params.tau)
        at = sample_categorical( probs )
            
    elif params.actpolicy == "egreedy"  :
        at = egreedy(m.Q, st, params.epsilon)
    
    return at

"""==============================================================================================================="""
