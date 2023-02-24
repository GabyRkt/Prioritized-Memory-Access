import numpy as np
from typing_extensions import ParamSpecArgs
from typing import Union
from maze import LinearTrack, OpenField
from parameters import Parameters
from mazemdp.toolbox import softmax, egreedy, egreedy_loc, sample_categorical
import matplotlib.pyplot as plt


# /!\ 

# The library from which we built our maze, SimpleMazeMDP, 
# doesn't consider a wall as a state, and therefore nb_states does account for walls,
# which differs from Mattar & Daw's code, where they do consider walls as states

# How this changes the code :
# ex : [ ][ ][Wall][ ][Reward]
# - Mattar's code consider the index of Reward to be 4
# - we consider the index of Reward to be 3


"""==============================================================================================================="""

def run_pre_explore(m : Union[LinearTrack,OpenField]) :
    """ Function that updates the transition matrix m.T from the first time

        Arguments:
            m -- Union[LinearTrack,OpenField] from maze.py
        
        Returns:      
    """
   
    # letting the agent try every action at each state, to build a transition matrix
    for s in range(0, m.nb_states) :
        for a in range(0,4) :
            if (s not in m.last_states) and (s != m.nb_states-1): # don't explore last state and well
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

    for last_state in m.last_states :
        
        # if OPEN FIELD
        if params.start_rand :
            m.T[last_state,:] = 0 # at first, put all transitions from last_state to 0

            # get a list of all index of valid start states : i.e not last_states or well
            l_valid_states = [ i for i in range(m.nb_states) if ( (i not in m.last_states) and (i!=m.nb_states-1) )  ]

            # transitions from goal to all possible start states have the same probability
            for valid_state in l_valid_states :
                 m.T[last_state,valid_state] = 1/len(l_valid_states)
        
        # if LINEAR TRACK
        else :
            m.T[last_state,:] = 0  # at first, put all transitions from last_state to 0
            # Top-Track Last State ==> Bottom-Track Start State
            if last_state == 18 :
                m.T[last_state,19] = 1
            # Bottom-Track Last State ==> Top-Track Start State
            elif last_state == 1 :
                m.T[last_state,0] = 1

    return

"""==============================================================================================================="""

def get_action(st, m: Union[LinearTrack,OpenField], params : Parameters) :

    if params.actpolicy == "softmax" :
        probs = softmax(m.Q, st, params.tau)
        at = sample_categorical( probs )
            
    elif params.actpolicy == "egreedy"  :
        at = egreedy(m.Q, st, params.epsilon)
    
    return at

"""==============================================================================================================="""

def update_transition_n_experience(st,at,r,stp1, m:Union[LinearTrack,OpenField], params:Parameters) :

    targVec = np.zeros( (1, m.nb_states) )
    targVec[0][stp1] = 1
    m.T[st,:] = m.T[st,:] + params.Talpha * ( targVec - m.T[st,:] ) # shift T-matrix towards targvec (?) => needs explanation

    m.listExp = np.append( m.listExp , [[st, at, r, stp1]], axis=0) # update the list of experiences  
    m.exp_LastStp1[st,at] = stp1 # update the list of last stp1 obtained with (st,at)
    m.exp_LastR[st,at] = r # update the list of last reward obtained with (st,at)

    return

"""==============================================================================================================="""

def update_q_table(st,at,r,stp1, m:Union[LinearTrack,OpenField], params:Parameters) :

    m.etr[st,:] = 0
    m.etr[st,at] = 1
    delta = r + params.gamma * np.max( m.Q[stp1] ) - m.Q[st,at]
    m.Q = m.Q + delta * m.etr
    m.etr = m.etr * params.lmbda * params.gamma

    return

"""==============================================================================================================="""

def create_plan_exp( m : Union[LinearTrack,OpenField], params : Parameters ) :
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

def get_gain (Q, planExp, params) :
    # planExp = [ step1, step3, ....]
    gain = []
    gain_matrix = np.empty(Q.shape)
    gain_matrix.fill(np.nan)
  

    for i in range(len(planExp)) :

        this_exp = planExp[i] #(st,a,r,stp1)

        if len(this_exp.shape) == 1:
            this_exp = np.expand_dims(this_exp, axis=0)

        gain.append(np.repeat(np.nan, this_exp.shape[0]))
                
        for j in range(this_exp.shape[0]):
                    
            #Q_mean = np.copy(self.Q[int(this_exp[j, 0])])
            Q_mean = np.copy(Q)

            Qpre = Q_mean  # NOT USING THIS??
            
            # Policy BEFORE backup

            #pA_pre = self.get_act_probs(Q_mean)
            pA_pre = softmax(Q_mean, int(this_exp[j, 0]), params.tau)

            # Value of state stp1
            stp1i = int(this_exp[-1, 3])

            #stp1_value = np.sum(np.multiply(Q[stp1i], self.get_act_probs(self.Q[stp1i])))
            stp1_value = np.sum(np.multiply( Q[stp1i], softmax(Q, stp1i, params.tau) ))               

            act_taken = int(this_exp[j, 1])
            steps_to_end = this_exp.shape[0] - (j + 1)

            rew = np.dot(np.power(params.gamma, np.arange(0, steps_to_end + 1)), this_exp[j:, 2])
            Q_target = rew + np.power(params.gamma, steps_to_end + 1) * stp1_value
            
            Q_mean[int(this_exp[j, 0]),act_taken] += params.alpha * (Q_target - Q_mean[int(this_exp[j, 0]),act_taken])

            # policy AFTER backup
            #pA_post = self.get_act_probs(Q_mean)
            pA_post = softmax(Q_mean, int(this_exp[j, 0]) , params.tau)
            
            # calculate gain
            EV_pre = np.sum(np.multiply(pA_pre, Q_mean[int(this_exp[j, 0])]))
            EV_post = np.sum(np.multiply(pA_post, Q_mean[int(this_exp[j, 0])]))
            
            gain[i][j] = EV_post - EV_pre

            
            Qpost = Q_mean  

            # Save on gain[s, a]
            sti = int(this_exp[j, 0])
            if np.isnan(gain_matrix[sti, act_taken]):
                gain_matrix[sti, act_taken] = gain[i][j]
            else:
                gain_matrix[sti, act_taken] = max(gain_matrix[sti, act_taken], gain[i][j])


    return gain, gain_matrix

"""==============================================================================================================="""

def get_need(sti, T, planExp, params) :
    need = []

    if params.onlineVSoffline == "online" :
        # there is a sti
        # Calculate the successor representation of the current state
        SR = np.linalg.inv(np.eye(len(T)) - params.gamma * T) # (I - gammaT)^-1
        SR_or_SD = SR[sti] 
  
    elif params.onlineVSoffline == "offline" :
        # The agent is asleep, there is not sti
        # Calculate eigenvectors and eigenvalues
        SR_or_SD = 0

    else :
        print("Error get_need : params.onlineVSoffline unknown")

    # Calculate the need term for each episode and each step
    for i_step in range (len(planExp)) :
        this_exp = planExp[i_step] #(st,a,r,stp1)
        if len(this_exp.shape) == 1:
                this_exp = np.expand_dims(this_exp, axis=0)
        need.append(np.repeat(np.nan, this_exp.shape[0]))
        for j in range(this_exp.shape[0]):
            need[i_step][j] = SR_or_SD[int(this_exp[j, 0])]
    
    return need

"""==============================================================================================================="""

def run_simulation(m : Union[LinearTrack,OpenField], params : Parameters) :

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
                EVBthresh = min(opportCost , params.EVBthresh)

                if max(EVB) > EVBthresh :
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
