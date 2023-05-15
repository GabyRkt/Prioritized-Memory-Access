from prio_replay.maze import LinearTrack, OpenField
from prio_replay.parameters import Parameters
from prio_replay.logger import Logger

from prio_replay.evb import get_gain, get_need, get_maxEVB_idx, calculate_evb
from prio_replay.q_learning import update_q_table, update_q_wplan, get_action
from prio_replay.transition_handler import run_pre_explore, add_goal2start, update_transition_n_experience
from prio_replay.planExp import create_plan_exp, expand_plan_exp, update_planning_backups

import matplotlib.pyplot as plt
import numpy as np


def plot_fig5g() :

    m = LinearTrack()
    params = Parameters()
    params.start_rand = False
    params.actpolicy = "softmax"
    params.tau = 0.2

    
    # IGNORE THIS : variables with no impact on the code
    # used to get certain functions working, has no bearing on the figure
    ep_i = 0 ; step_i = 0; log = Logger(); prev_s=0; prev_stp1=0
    log.nb_backups_per_state = [ [0] * m.nb_states ] * params.MAX_N_EPISODES
    log.forward_per_state =  [ [0] * m.nb_states ] 
    log.backward_per_state = [0] * m.nb_states
    log.nbvisits_per_state = [0] * m.nb_states

    
    # RUN SIMULATION BEFORE SHOCK : LET THE AGENT EXPLORE AND PLAN WITHOUT REWARD
    
    # move the agent to the first starting state
    st = 0
    run_pre_explore(m)
    add_goal2start(m,params)

    planning_backups_pre = np.empty( (0,5) )
    p = 1

    while p <= params.Nplan :

                # create a matrix that records the reward r and next-state stp1 of each (st,at) 
                planExp = create_plan_exp(m,params)
                planExp = list(planExp)

                if params.expandFurther and planning_backups_pre.shape[0] > 0 :
                    expand_plan_exp(planning_backups_pre, planExp, m, params)

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

                    # Add the updated planExp to planning_backups 
                    planning_backups_pre = update_planning_backups(planning_backups_pre, plan_exp_arr_max)
                
                p += 1

    # RUN SIMULATION BEFORE SHOCK : LET THE AGENT EXPLORE AND PLAN WITHOUT REWARD

    # implant memory of a shock ( yes this is how they did it )
    m.exp_LastR[16,2] = -1 
    m.exp_LastStp1[16,2] = 18

    need_pre = np.linalg.inv(np.eye(len(m.T)) - params.gamma * m.T)[0][16]
    gain_pre = saGain[16][2]

    need_post = 0
    gain_post = 0

    planning_backups_post = np.empty( (0,5) )
    p = 1

    # run planning again
    while p <= params.Nplan :

                # create a matrix that records the reward r and next-state stp1 of each (st,at) 
                planExp = create_plan_exp(m,params)
                planExp = list(planExp)

                if params.expandFurther and planning_backups_post.shape[0] > 0 :
                    expand_plan_exp(planning_backups_post, planExp, m, params)

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
                    
                    if (plan_exp_arr_max[0][0] == 16.0) and (plan_exp_arr_max[0][1] == 2):
                        need_post = need[maxEVB_idx]
                        gain_post = saGain[16][2]
                    
                    #Update q_values using plan_exp_arr_max
                    prev_s , prev_stp1 = update_q_wplan(ep_i, st, p, log, step_i, prev_s, prev_stp1, plan_exp_arr_max, m, params)

                    # Add the updated planExp to planning_backups 
                    planning_backups_post = update_planning_backups(planning_backups_post, plan_exp_arr_max)
                
                p += 1


    count_pre = np.count_nonzero(planning_backups_pre == 16)
    count_post = np.count_nonzero(planning_backups_post == 16)

    fig, ax = plt.subplots(3)

    x = ["before shock","after shock"]
    
    y = [ count_pre , count_post ]
    ax[0].bar(x,y, width= 0.3, color="green")
    ax[0].set_title("Activation Probality before/after shock delivery ")

    y_need = [ need_pre , need_post ]
    ax[1].bar(x,y_need, width= 0.3, color="blue")
    ax[1].set_title("Need Term before/after shock delivery ")

    y_gain = [ gain_pre , gain_post ]
    ax[2].bar(x,y_gain, width= 0.3, color="yellow")
    ax[2].set_title("Gain Term before/after shock delivery ")

    plt.show()


    return


plot_fig5g()