import numpy as np
from typing_extensions import ParamSpecArgs
from typing import Union
from maze import LinearTrack, OpenField
from parameters import Parameters
from mazemdp.toolbox import softmax, egreedy, egreedy_loc, sample_categorical

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
    # create a matrix with shape :
    #    [ [ s[0] a[0] last_r[0,0] last_stp1[0,0] ] ]
    #    [ [ s[1] a[0] last_r[1,0] last_stp1[1,0] ] ]
    #                  ....
    #    [ [ s[0] a[1] last_r[0,1] last_stp1[0,1] ] ]
    #               ..........

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

def sim2(m : Union[LinearTrack,OpenField], params : Parameters) :

    m.reInit()
    m.mdp.timeout = params.MAX_N_STEPS

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
        st = m.reset()

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

            p = 20 # planning step counter

            if params.planOnlyAtGorS : # only plan when agent is in a start/last state...
                if not ( (st in m.maze.last_states) or (step_i == 0) ):
                    p = np.Inf
            
            # ...but don't if this is the 1st episode and the reward hasn't been found 
            if ( (r == 0) and (ep_i == 0) ) :
                p = np.Inf

            # pre-allocating planning variables
            planning_backups = np.empty( (0,5) )

            while p <= params.Nplan :
                p += 1
                # create a matrix that records the reward r and next-state stp1 of each (st,at) 
                planExp = create_plan_exp(m,params)
                planExp = list(planExp)

                if params.expandFurther and planning_backups.shape[0] > 0 :
                    ...
                
                

            st = stp1
            step_i = step_i + 1


    return


p = Parameters()
m = OpenField()

sim2(m,p)