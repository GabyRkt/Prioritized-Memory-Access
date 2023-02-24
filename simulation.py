import numpy as np
from typing_extensions import ParamSpecArgs
from typing import Union
from maze import LinearTrack, OpenField
from parameters import Parameters
from mazemdp.toolbox import softmax, egreedy, egreedy_loc, sample_categorical
import matplotlib.pyplot as plt
from evb import *
from q_learning import *
from transition_handler import *


# /!\ 

# The library from which we built our maze, SimpleMazeMDP, 
# doesn't consider a wall as a state, and therefore nb_states does account for walls,
# which differs from Mattar & Daw's code, where they do consider walls as states

# How this changes the code :
# ex : [ ][ ][Wall][ ][Reward]
# - Mattar's code consider the index of Reward to be 4
# - we consider the index of Reward to be 3


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
        for s in range(m.nb_states-1) :
            exp = np.array( [ s, a, m.exp_LastR[s,a], m.exp_LastStp1[s,a] ] ) 
            planExp = np.append( planExp, [exp], axis=0 )
    
    if params.remove_samestate : # remove all experiences that goes to the same states (i.e walls)
        planExp = planExp[planExp[:, 0] != planExp[:, 3]]
    
    planExp = planExp[ np.invert( np.isnan(planExp).any(axis=1) ) ] # remove all experiences with NaNs in it , we dont need this theoretically

    return planExp

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
                if not ( (st in m.maze.last_states) or (step_i == 0) ):
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
                    
                    seqStart = np.argwhere(planning_backups[:, 4] == 1)[-1][0]
                    seqSoFar = planning_backups[seqStart:, 0:4]
                    
                    sn = int( seqSoFar[-1, 3] )  # Final state reached in the last planning st

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
                
                
                # === Gain term ===
                [ gain , saGain ] = get_gain(m.Q,planExp,params)
                
                # === Need term ===
                need = get_need(st, m.T, planExp, params)
                  

                # === EVB ===  GET MAX EVB 
                EVB = np.empty( len(planExp ) )
                EVB.fill(np.nan)

                for i in range( len(planExp) ) :
                    EVB[i] = np.sum( need[i][0] * max( gain[i][-1], params.baselineGain ) )
                
                opportCost = np.nanmean( m.listExp[:,2] )
                EVBthreshold = min(opportCost , params.EVBthreshold)

                if max(EVB) > EVBthreshold :
                    maxEVB_idx = np.argwhere(EVB == max(EVB))

                    if len(maxEVB_idx) > 1 :
                        n_steps = np.array([arr.shape[0] if len(arr.shape) > 1 else 1 for arr in planExp])
                        maxEVB_idx = maxEVB_idx[n_steps[maxEVB_idx] == min(n_steps[maxEVB_idx])]
                        if len(maxEVB_idx) > 1 :
                            maxEVB_idx = maxEVB_idx[np.random.randint(len(maxEVB_idx))]  
                    else:
                        maxEVB_idx = maxEVB_idx[0][0]
                
                    plan_exp_arr = np.array(planExp, dtype=object)
                                        
                    if len(plan_exp_arr[maxEVB_idx].shape) == 1:
                        plan_exp_arr_max = np.expand_dims(plan_exp_arr[maxEVB_idx], axis=0)
                    else:
                        plan_exp_arr_max = np.expand_dims(plan_exp_arr[maxEVB_idx][-1], axis=0)
                                        
                    for n in range(plan_exp_arr_max.shape[0]):
                        # Retrieve information from this experience
                        tmp = plan_exp_arr_max[n]
                        if len(plan_exp_arr_max[n]) == 1 :
                            tmp = plan_exp_arr_max[n][0]
                            

                        s_plan = int(tmp[0])
                        a_plan = int(tmp[1])
                        
                        # Individual rewards from this step to end of trajectory
                        rew_to_end = plan_exp_arr_max[n:][:, 2]
                        # Notice the use of '-1' instead of 'n', meaning that stp1_plan is the final state of the
                        # trajectory
                        stp1_plan = int(plan_exp_arr_max[-1][3])

                        # Discounted cumulative reward from this step to end of trajectory
                        n_plan = np.size(rew_to_end)
                        r_plan = np.dot(np.power(params.gamma, np.arange(0, n_plan)), rew_to_end)

                        # ADD PLAN Q_LEARNING UPDATES TO Q_LEARNING FUNCTION
                        stp1_value = np.sum(np.multiply(m.Q[stp1_plan], softmax(m.Q,stp1_plan,params.tau)  ))
                  
                        Q_target = r_plan + (params.gamma ** n_plan) * stp1_value
                      
                        m.Q[s_plan, a_plan] += params.alpha * (Q_target - m.Q[s_plan, a_plan])


                    if planning_backups.shape[0] > 0:
                        planning_backups = np.vstack([planning_backups, np.append(plan_exp_arr_max, plan_exp_arr_max.shape[0])])
                    elif planning_backups.shape[0] == 0:
                        planning_backups = np.append(plan_exp_arr_max,plan_exp_arr_max.shape[0]).reshape(1, planning_backups.shape[1])
                    else:
                        err_msg = 'planning_backups does not have the correct shape. It is {} but should have a length equal to 1 or 2, e.g. (5,) or (2, 5)'.format(planning_backups.shape)
                        raise ValueError(err_msg)
                
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


p = Parameters()
m = OpenField()

p.Nplan = 20

test_list = run_simulation(m,p)

print("\n\n", test_list[0])

plt.plot(test_list)
plt.show()
