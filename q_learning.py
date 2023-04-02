import numpy as np
from typing_extensions import ParamSpecArgs
from typing import Union
from maze import LinearTrack, OpenField
from parameters import Parameters
from mazemdp.toolbox import softmax, egreedy, egreedy_loc, sample_categorical
import matplotlib.pyplot as plt
import seaborn as sns


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

def update_q_wplan (st, p, log, step_i, prev_s, prev_stp1, plan_exp_arr_max, m, params):
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

        if False:
            SR = np.linalg.inv(np.eye(len(m.T)) - params.gamma * m.T)
            b = [ max(i) for i in m.Q ]
            q_after = [
                [ b[0]  , b[2]  , b[4]  , b[6]  , b[8]  , b[10] , b[12] , b[14] , b[16] , b[18]  ],
                [ np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN ],
                [ b[1]  , b[3]  , b[5]  , b[7]  , b[9]  , b[11] , b[13] , b[15] , b[17] , b[19]  ],
            ]
            b = [ ' ' for i in range(m.nb_states) ]
            if a_plan == 0 :
                b[s_plan] = "^"
            elif a_plan == 1 :
                b[s_plan] = "v"
            elif a_plan == 2 :
                b[s_plan] = ">"
            else :
                b[s_plan] = "<" 
            
            b[st] = "o"
            
  
            gain_annot = [
                [ b[0]  , b[2]  , b[4]  , b[6]  , b[8]  , b[10] , b[12] , b[14] , b[16] , b[18]  ],
                [ np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN ],
                [ b[1]  , b[3]  , b[5]  , b[7]  , b[9]  , b[11] , b[13] , b[15] , b[17] , b[19]  ],
            ]
        
            # ax2 = sns.heatmap(q_after, fmt="", vmin=0, vmax=1, annot = gain_annot,cmap="Blues_r",square=True, yticklabels=False, xticklabels=False)
            # plt.plot()
            # plt.show(block=False)
            # plt.pause(1)
            # plt.close()
            # b = [ (max(i)*5).round(3) for i in saGain ]
            # gain_annot = [
            #     [ b[0]  , b[2]  , b[4]  , b[6]  , b[8]  , b[10] , b[12] , b[14] , b[16] , b[18]  ],
            #     [ np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN ],
            #     [ b[1]  , b[3]  , b[5]  , b[7]  , b[9]  , b[11] , b[13] , b[15] , b[17] , b[19]  ],
            # ]
        
            ax2 = sns.heatmap(q_after, fmt="", vmin=0, vmax=1, annot = gain_annot,cmap="Blues_r",square=True, yticklabels=False, xticklabels=False)
            plt.plot()
            plt.show(block= False)
            plt.pause(0.2)
            plt.close()


        # Discounted cumulative reward from this step to end of trajectory
        n_plan = np.size(rew_to_end)
        r_plan = np.dot(np.power(params.gamma, np.arange(0, n_plan)), rew_to_end)

        # Add plan and q_learning updates to q_learning function
        stp1_value = np.max(m.Q[stp1_plan])
        Q_target = r_plan + (params.gamma ** n_plan) * stp1_value

        # Update Q-value for this state-action pair 
        m.Q[s_plan, a_plan] = m.Q[s_plan, a_plan] + params.alpha * (Q_target - m.Q[s_plan, a_plan])

        if (p > 1) :
            if (s_plan != prev_s) and (st % 2) == (s_plan % 2):
                
                if step_i == 0 :
            
                    if s_plan == prev_stp1  :
                        log.nbStep_forwardReplay_forward += 1
                    
                    if stp1_plan == prev_s :
                        log.nbStep_backwardReplay_forward += 1

                else :
                    
                    if s_plan == prev_stp1  :
                        log.nbStep_forwardReplay_backward += 1
                    
                    if stp1_plan == prev_s :
                        log.nbStep_backwardReplay_backward += 1

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
