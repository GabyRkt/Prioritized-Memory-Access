
import numpy as np


"""==============================================================================================================="""

def softmax(q, x, tau):
    # This function taken from the mazemdp.toolbox library, but with a small modification to not round down values

    # Returns a soft-max probability distribution over actions
    # Inputs :
    # - Q : a Q-function represented as a nX times nU matrix
    # - x : the state for which we want the soft-max distribution
    # - tau : temperature parameter of the soft-max distribution
    # Note that tau can be set to 0 because numpy can deal with the division by 0
    # Output :
    # - p : probability of each action according to the soft-max distribution

    p = np.zeros((len(q[x])))
    sump = 0
    for i in range(len(p)):
        p[i] = np.exp((q[x, i] / tau))
        sump += p[i]

    p = p / sump
    return p


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

        gain.append(np.repeat(np.nan, 1))
                
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

    # not sleeping (there is a sti)
    # Calculate the successor representation of the current state
    SR = np.linalg.inv(np.eye(len(T)) - params.gamma * T) # (I - gammaT)^-1
    SRi = SR[sti] 

    # Calculate the need term for each episode and each step
    for i_step in range (len(planExp)) :
        this_step = planExp[i_step] #(st,a,r,stp1)
        need.append(np.repeat(np.nan, len(this_step)))

        need[i_step][0] = SRi[int(this_step[0])]

    return need

"""==============================================================================================================="""

from typing_extensions import ParamSpecArgs
from typing import Union
from maze import LinearTrack, OpenField
from parameters import Parameters
from mazemdp.toolbox import egreedy, egreedy_loc, sample_categorical


def run_simulation( m : Union[LinearTrack,OpenField], params : Parameters, render: bool = False) :
  
  if render:
    m.mdp.new_render("Simulation: Prioritized Replay")
  
  m.mdp.timeout = params.MAX_N_STEPS 

  q_list = dict()

  #============================ PRE-EXPLORATION =============================#

  if params.preExplore :
    
    for s in range(0, m.nb_states) :
      for a in range(0,4) :
        if (s not in m.walls) and (s not in m.last_states) and (s != m.nb_states-1) : # don't explore walls, last state and well
          m.mdp.current_state = s
          [stp1, _, _, _] = m.mdp.step(a) 
          m.listExp = np.append( m.listExp , [[s, a, 0, stp1]], axis=0) # update the list of experiences  
          m.exp_LastStp1[s,a] = stp1 # update the list of last stp1 obtained with (st,at)
          m.exp_LastR[s,a] = 0 # update the list of last reward obtained with (st,at)
          m.T[s,stp1] = m.T[s,stp1] + 1
    
  #choose a random starting state 
  st = m.reset()

    #============================= EXPLORATION ================================#
    
  for nbEpisode in range(params.MAX_N_EPISODES):

    done = False

    #LOOP FOR N STEPS OR REWARD FOUND
    nbSteps = 0
    while not done:

      tmp = np.copy(m.Q)
      q_list[nbEpisode] = tmp

      if render:
        m.mdp.render( m.Q, m.Q.argmax(axis=1) )
        
      # === Action Selection === GET ACTION
      if params.actpolicy == "softmax" :
        probs = softmax(m.Q, st, params.tau)
        at = sample_categorical( probs )
      elif params.actpolicy == "egreedy"  :
        at = egreedy(m.Q, st, params.epsilon)

      # === Perform Action ===
      [stp1, r, done, _] = m.mdp.step(at)
        
      # === Update Transition Matrix and Experience List ===
      targVec = np.zeros( (1, m.nb_states) )
      targVec[0][stp1] = 1
      m.T[st,:] = m.T[st,:] + params.Talpha * ( targVec - m.T[st,:] ) # shift T-matrix towards targvec (?) => needs explanation

      m.listExp = np.append( m.listExp , [[st, at, r, stp1]], axis=0) # update the list of experiences  
      m.exp_LastStp1[st,at] = stp1 # update the list of last stp1 obtained with (st,at)
      m.exp_LastR[st,at] = r # update the list of last reward obtained with (st,at)

      # === Update Q-table with an eligibility trace === Q LEARNING 
      m.etr[st,:] = 0
      m.etr[st,at] = 1

      delta = r + params.gamma * np.max( m.Q[stp1] ) - m.Q[st,at]
      m.Q = m.Q + delta * m.etr
      
      m.etr = m.etr * params.lmbda * params.gamma
        
    #============================== PLANNING  ==================================#  

      # === Planning prep ===  PREP PLANNING 

      p = 1 # Planning step counter

      if params.planOnlyAtGorS :
        if not ( (st in m.maze.last_states) or (nbSteps == 0) ):
          p = np.Inf
      
      if ( (r == 0) and (nbEpisode == 0) ) :
        p = np.Inf

      # === Pre-allocate variables ===
      
      planning_backups = np.empty( (0,5) )
      
      
      while p <= params.Nplan :
        print(p)
        p = p + 1
        # === Create a list of 1-step backups based on 1-step models === CREATE PLAN EXP
        
        planExp = np.empty( (0,4) )

        for a in range(m.action_space.n) :
          for s in range(m.nb_states-1) :
            exp = np.array( [ s, a, m.exp_LastR[s,a], m.exp_LastStp1[s,a] ] ) 
            planExp = np.append( planExp, [exp], axis=0 )
        
        if params.remove_samestate : # remove all experiences that goes to the same states (i.e walls)
          planExp[planExp[:, 0] != planExp[:, 3]]
        
        planExp = planExp[ np.invert( np.isnan(planExp).any(axis=1) ) ] # remove all experiences with NaNs in it , we dont need this theoretically

        # === Expand previous backup with one extra action ===   EXPAND PREVIOUS BACKUP
        if params.expandFurther and np.size(planning_backups,0) > 0 :
          
          seqStart = np.argwhere(planning_backups[:, 4] == 1)[-1]
          seqSoFar = planning_backups[seqStart[0]:, 0:4]
          sn = int( seqSoFar[-1, 3] )  # Final state reached in the last planning st

          if params.onlineVSoffline == "online" : # agent is awake 
            
            probs = softmax(m.Q, sn, params.tau)
          
          else : # agent is asleep 
            probs = np.zeros( np.size(m.Q[sn]) )
            probs[ m.Q[sn] == max( m.Q[sn] ) ] =  1 / ( sum(m.Q[sn]) == max(m.Q[sn]) ) 

          an = sample_categorical( probs )
          snp1 = m.exp_LastStp1[sn,an]
          rn = m.exp_LastR[sn,an]

          step_isNaN = np.isnan( m.exp_LastStp1[sn,an] ) or np.isnan( m.exp_LastR[sn,an] )
          step_isRepeated = np.isin( snp1 , [ seqSoFar[:, 0], seqSoFar[:, 3] ] )

          if (not step_isNaN) and (params.allowLoops or (not step_isRepeated)) :
            expanded_exp = np.array( [sn, an, rn, snp1] )
            seqUpdated = np.append( seqSoFar, [expanded_exp], axis=0 )
            planExp = np.append(planExp, seqUpdated)
            

        
        # === Gain term ===
        [ gain , saGain ] = get_gain(m.Q,planExp,params)
        
        # === Need term ===
        need = get_need(st, m.T, planExp, params)

        if nbEpisode == 45 and not done :
          m.needMat[st,st] = need[-1][0] 
          

        # === EVB ===  GET MAX EVB 
        EVB = np.empty( planExp.shape[0] )
        EVB.fill(np.nan)

        for i in range(planExp.shape[0]) :
          EVB[i] = np.sum( need[i][-1] * max( gain[i][-1], params.baselineGain ) )
        
        opportCost = np.nanmean( m.listExp[:,2] )
        EVBthresh = min(opportCost , params.EVBthresh)

        if max(EVB) > EVBthresh :
          maxEVB_idx = np.argwhere(EVB == max(EVB))

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

              # ADD PLAN Q_LEARNING UPDATES TO Q_LEARNING FUNCTION
              stp1_value = np.sum(np.multiply(m.Q[stp1_plan], softmax(m.Q,stp1_plan,params.tau)  ))
          
              Q_target = r_plan + (params.gamma ** n_plan) * stp1_value
              
              m.Q[s_plan, a_plan] += params.alpha * (Q_target - m.Q[s_plan, a_plan])

    #============================== COMPLETE STEP ==================================#

      # === Move from st to stp1 ===
      st = stp1
      nbSteps = nbSteps + 1

      if done :
        st = m.reset()

        if (nbSteps < params.MAX_N_STEPS) and params.Tgoal2start : 
          targVec = np.zeros( (1, m.nb_states) )
          targVec[0][st] = 1
          m.T[stp1,:] = m.T[stp1,:] + params.Talpha * ( targVec - m.T[stp1,:] ) # shift T-matrix towards targvec (?) => needs explanation
          m.listExp = np.append( m.listExp , [[stp1, np.NaN, np.NaN, st]], axis=0)

  if render:
      # Show the final policy
      m.mdp.current_state = 0
      m.mdp.render(m.Q, np.argmax(m.Q, axis = 1), title="Simulation: Prioritized Replay")
    
  return q_list



"""==============================================================================================================="""



def evaluate(m : Union[OpenField,LinearTrack] , q_list, params : Parameters, render = False) :

  m.mdp.timeout = params.MAX_N_STEPS
  if render :
    m.mdp.new_render("Simulation: Prioritized Replay")

  list_steps = []
  
  for nbEpisode in range(20) :
    done = False
    st = m.reset()

    nb_steps = 0
    
    while not done :
      curr_q = q_list[nbEpisode]
      if render :
        m.mdp.render( curr_q, curr_q.argmax(axis=1) )

      at = egreedy(curr_q, st, 0.1)
      [ stp1, r , done, _ ] = m.mdp.step(at)

      st = stp1
      nb_steps += 1
    
    list_steps.append(nb_steps)

  return list_steps

