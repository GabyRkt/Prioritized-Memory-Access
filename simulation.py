import numpy as np
from typing_extensions import ParamSpecArgs
from typing import Union
from maze import LinearTrack, OpenField
from parameters import Parameters
from mazemdp.toolbox import softmax, egreedy, egreedy_loc, sample_categorical
from evb import get_gain, get_need, get_maxEVB_idx, calculate_evb
from q_learning import update_q_table, update_q_wplan, get_action
from transition_handler import run_pre_explore, add_goal2start, update_transition_n_experience
from planExp import create_plan_exp, expand_plan_exp, update_planning_backups
from logger import Logger


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
    """ Calculates the need for a state depending on planExp and the current mode of the agent (offline or online)

        Arguments
        ----------
            m -- Union[LinearTrack,OpenField] from maze.py : class with the maze and the agent
            params -- Parameters from parameters.py : class with the settings of the current simulation 
        
        Returns
        ----------
            list_steps -- list of int : list of number of steps taken by the agent during the simulation for each episode
    """

    m.reInit()
    m.mdp.timeout = params.MAX_N_STEPS
    list_steps = []

    log = Logger()

    # [ PRE-EXPLORATION ]

    # Have the agent freely explore the maze without rewards to learn action consequences
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

            # Update Transition Matrix & Experience List with stp1 and r
            update_transition_n_experience(st,at,r,stp1, m, params)

            # Update Q-table : off-policy Q-learning using eligibility trace
            update_q_table(st,at,r,stp1, m, params)


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
                    update_q_wplan(st, p, log, step_i, plan_exp_arr_max, m, params)

                    # Add the updated planExp to planning_backups 
                    planning_backups = update_planning_backups(planning_backups, plan_exp_arr_max)
                
                p += 1
            #============================== COMPLETE STEP ==================================#

            # === Move from st to stp1 ===
            st = stp1
            step_i = step_i + 1

            if done :
                st = m.reset()

                if (step_i < params.MAX_N_STEPS) and params.Tgoal2start : 
                    targVec = np.zeros( (1, m.nb_states) )
                    targVec[0][st] = 1
                    m.T[stp1,:] = m.T[stp1,:] + params.Talpha * ( targVec - m.T[stp1,:] ) # shift T-matrix towards targvec (?) => needs explanation
                    m.listExp = np.append( m.listExp , [[stp1, np.NaN, np.NaN, st]], axis=0)

        list_steps.append(step_i)

    return list_steps


