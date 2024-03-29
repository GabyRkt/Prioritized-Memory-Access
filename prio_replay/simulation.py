import numpy as np
from typing_extensions import ParamSpecArgs
from typing import Union
from mazemdp.toolbox import softmax, egreedy, egreedy_loc, sample_categorical
import random
import matplotlib.pyplot as plt
import scipy 
import seaborn as sns

from prio_replay.maze import LinearTrack, OpenField, Tmaze
from prio_replay.parameters import Parameters
from prio_replay.evb import get_gain, get_need, get_maxEVB_idx, calculate_evb
from prio_replay.q_learning import update_q_table, update_q_wplan, get_action
from prio_replay.transition_handler import run_pre_explore, add_goal2start, update_transition_n_experience
from prio_replay.planExp import create_plan_exp, expand_plan_exp, update_planning_backups
from prio_replay.logger import Logger



# /!\ 

# The library from which we built our maze, SimpleMazeMDP, 
# doesn't consider a wall as a state, and therefore nb_states does account for walls,
# which differs from Mattar & Daw's code, where they do consider walls as states

# How this changes the code :
# ex : [ ][ ][Wall][ ][Reward]
# - Mattar's code consider the index of Reward to be 4
# - we consider the index of Reward to be 3


"""==============================================================================================================="""


def run_simulation(m : Union[LinearTrack,OpenField], params : Parameters) :
    """ 
        Arguments
        ----------
            m -- Union[LinearTrack,OpenField] from maze.py : class with the maze and the agent
            params -- Parameters from parameters.py : class with the settings of the current simulation 

    """
    m.reInit()
    m.mdp.timeout = params.MAX_N_STEPS

    # initializing logger to store useful data
    log = Logger()
    
    for i in range(params.MAX_N_EPISODES) : 
        f = []; b = []; s = []

        log.forward_per_state.append(f)
        log.backward_per_state.append(b)
        log.steps_per_episode.append(s)

        backups = [0] * m.nb_states 

        log.nb_backups_per_state.append(backups)
        
    log.nbvisits_per_state = [0] * m.nb_states

    
    # [ PRE-EXPLORATION ]

    # Have the agent freely explore the maze without rewards to learn action consequences !
    if params.preExplore : 
        run_pre_explore(m)

    
    # Add transitions from goal states to start states : this loops the need value? => needs explanation
    if params.Tgoal2start: 
        add_goal2start(m, params)


    #  [ EXPLORATION ]

    # choose a starting state
    st = m.reset()


    for ep_i in range(params.MAX_N_EPISODES) :
        
        step_i = 0 # step counter
        done = False # this will be True when the agent finds a reward, or when MAX_N_STEPS has been reached
        print("running : [ episode "+str(ep_i)+ " ]",end='\r')

        while not done:  
            

            # [ CLASSIC Q-LEARNING ]

            # Action Selection with softmax or epsilon-greedy, using Q-Learning
            at = get_action(st, m, params)
            # Perform Action : st , at  
            [stp1, r, done, _] = m.mdp.step(at)

            # saving some data
            log.nbvisits_per_state[st] += 1

            # setting reward
            if params.changeR and r :
                if params.x4:
                    if ep_i in ([x for x in range (2, 50, 4)]+[x for x in range (3, 50, 4)]) :
                        r = 4
                        log.event_episode.append(ep_i)
                elif params.x0 :
                    if ep_i in ([x for x in range (2, 50, 4)]+[x for x in range (3, 50, 4)]):  
                        r = 0    
                        log.event_episode.append(ep_i)


            # add gaussian noise to r if reward is found
            if r :
                r = random.gauss(r,params.sigma)

            # Update Transition Matrix & Experience List with stp1 and r
            update_transition_n_experience(st,at,r,stp1, m, params)

            # Update Q-table : off-policy Q-learning using eligibility trace
            update_q_table(st,at,r,stp1, m, params)
            log.nb_replay_per_ep[ep_i] += 1

            if ep_i > 250 :
                a = [ max(elem) for elem in m.Q ]
                need_b4 = [
                    [ a[0]  , a[2]  , a[4]  , a[6]  , a[8]  , a[10] , a[12] , a[14] , a[16] , a[18]  ],
                    [ np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN ],
                    [ a[1]  , a[3]  , a[5]  , a[7]  , a[9]  , a[11] , a[13] , a[15] , a[17] , a[19]  ],
                ]
                a = [ " " for elem in m.Q ]
                a[st] = str(at)
                need_b5 = [
                    [ a[0]  , a[2]  , a[4]  , a[6]  , a[8]  , a[10] , a[12] , a[14] , a[16] , a[18]  ],
                    [ np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN ],
                    [ a[1]  , a[3]  , a[5]  , a[7]  , a[9]  , a[11] , a[13] , a[15] , a[17] , a[19]  ],
                ]
                sns.heatmap(need_b4,annot=need_b5,fmt="")
                plt.show(block=False)
                plt.pause(1)
                plt.close()

            # [ PLANNING ]
            p = 1 # planning step counter

            if params.planOnlyAtGorS : # only plan when agent is in a start/last state...
                if not ( (stp1 in m.maze.last_states) or (step_i == 0) ):
                    p = np.Inf
            
            # ...but don't if this is the 1st episode and the reward hasn't been found 
            if ( (r == 0) and (ep_i == 0) ) :
                p = np.Inf

            # pre-allocating planning variables
            planning_backups = np.empty( (0,5) )
            prev_s = np.NaN
            prev_stp1 = np.NaN

            while p <= params.Nplan :

                # create a matrix that records the reward r and next-state stp1 of each (st,at) 
                planExp = create_plan_exp(m,params)
                planExp = list(planExp)

                if params.expandFurther and planning_backups.shape[0] > 0 :
                    expand_plan_exp(planning_backups, planExp, m, params)

                # === Gain term ===
                if params.allgain2one : # we set all gain to one to simulate Random Replay
                    gain = list(np.ones((len(planExp), 1)))
                else :
                    [ gain , saGain ] = get_gain(m.Q,planExp,params)


                # === Need term ===
                if params.allneed2one : # we set all need to one to simulate Random Replay
                    need = list(np.ones((len(planExp), 1)))
                else :
                    need = get_need(st, m.T, planExp, params)

                # === EVB ===  
                EVB = calculate_evb(planExp, gain, need, params)

                opportCost = np.nanmean( m.listExp[:,2] )
                EVBthreshold = min(opportCost , params.EVBthreshold)
    
                if max(EVB) > EVBthreshold :
                    maxEVB_idx = get_maxEVB_idx(EVB, planExp)
                    
                    plan_exp_arr = np.array(planExp, dtype=object)

                    if len(plan_exp_arr[maxEVB_idx].shape) == 1:
                        plan_exp_arr_max = np.expand_dims(plan_exp_arr[maxEVB_idx], axis=0)
                    else:
                        plan_exp_arr_max = np.expand_dims(plan_exp_arr[maxEVB_idx][-1], axis=0)

                    #Update q_values using plan_exp_arr_max
                    prev_s , prev_stp1 = update_q_wplan(ep_i, st, p, log, step_i, prev_s, prev_stp1, plan_exp_arr_max, m, params)

                    if p == 1 :
                        log.dist_agent_replay_start.append( (st,prev_s) )

                    # Add the updated planExp to planning_backups 
                    planning_backups = update_planning_backups(planning_backups, plan_exp_arr_max)
                    if planning_backups.shape[0] > 0:
                        log.replay_state[ep_i] = planning_backups[:,0]
                        log.replay_action[ep_i] = planning_backups[:,1]

                    if ep_i == params.MAX_N_EPISODES : 
                        log.replay_state[ep_i] = log.replay_state[:ep_i]
                        log.replay_action[ep_i] = log.replay_action[:ep_i]
                
                p += 1
            #============================== COMPLETE STEP ==================================#

            # === Move from st to stp1 ===
            st = stp1
            step_i = step_i + 1
            
            log.steps_per_episode[ep_i].append(st)

            

            if done :
                st = m.reset()

                if (step_i < params.MAX_N_STEPS) and params.Tgoal2start : 
                    targVec = np.zeros( (1, m.nb_states) )
                    targVec[0][st] = 1
                    m.T[stp1,:] = m.T[stp1,:] + params.Talpha * ( targVec - m.T[stp1,:] ) # shift T-matrix towards targvec (?) => needs explanation
                    m.listExp = np.append( m.listExp , [[stp1, np.NaN, np.NaN, st]], axis=0)

            

        log.nbStep.append(step_i)

    return log


