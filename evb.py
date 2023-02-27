import numpy as np
from typing_extensions import ParamSpecArgs
from typing import Union
from maze import LinearTrack, OpenField
from parameters import Parameters
from mazemdp.toolbox import softmax, egreedy, egreedy_loc, sample_categorical


"""==============================================================================================================="""

def get_gain (Q, planExp, params) :
    """ Calculates the gain of each state and each action based on the current Q-table and planExp

        Arguments
        ----------
            Q -- matrix ( state X action ) : the current Q-table
            planExp -- matrix ((state X action) X 4) : memory of last reward and next state obtained of each tuple (state, action)
            params -- Parameters from parameters.py : class with the settings of the current simulation 
        
        Returns
        ----------   
            gain -- list : list of the gain calulated for each experience in planExp
            gain_matrix -- matrix ( state X action) : the maximum gain calculated for each state and each action
    """
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
            stp1_value = np.max(Q[stp1i])                

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

def get_need(st, T, planExp, params) :
    """ Calculates the need for a state depending on planExp and the current mode of the agent (offline or online)

        Arguments
        ----------
            st -- int : the current state
            T : matrix ( state X state ) : matrix of transition
            planExp -- matrix ((state X action) X 4) : memory of last reward and next state obtained of each tuple (state, action)
            params -- Parameters from parameters.py : class with the settings of the current simulation 
        
        Returns
        ----------   
            need -- float : need calculated for the state in parameter 
    """

    need = []

    if params.onlineVSoffline == "online" :
        # there is a st
        # Calculate the successor representation of the current state
        SR = np.linalg.inv(np.eye(len(T)) - params.gamma * T) # (I - gammaT)^-1
        SR_or_SD = SR[st] 
  
    elif params.onlineVSoffline == "offline" :
        # The agent is asleep, there is not st
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

def calculate_evb(planExp, gain, need, params) :
    EVB = np.full((len(planExp)), np.nan)

    for i in range( len(planExp) ) :
        if len(planExp[i].shape) == 1:
            EVB[i] = need[i][-1] * max( gain[i], params.baselineGain ) 
        else :
            EVB[i] = 0
            for x in range(len(planExp[i])) :
                EVB[i] += need[i][-1] * max(gain[i][-1], params.baselineGain)
    
    return EVB