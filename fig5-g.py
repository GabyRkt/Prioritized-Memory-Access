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
import random
import matplotlib.pyplot as plt

def plot_fig5g() :

    m = LinearTrack()
    params = Parameters()
    params.start_rand = False
    params.actpolicy = "softmax"
    params.tau = 0.2

    
    # IGNORE THIS : variables with no impact on the code
    # used to get certain functions working, has no bearing on the figure
    ep_i = 0 ; step_i = 0; log = Logger(); prev_s=0; prev_stp1=0

    
    
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

                    #Update q_values using plan_exp_arr_max
                    prev_s , prev_stp1 = update_q_wplan(ep_i, st, p, log, step_i, prev_s, prev_stp1, plan_exp_arr_max, m, params)

                    # Add the updated planExp to planning_backups 
                    planning_backups_post = update_planning_backups(planning_backups_post, plan_exp_arr_max)
                
                p += 1



    count_pre = np.count_nonzero(planning_backups_pre == 16)
    count_post = np.count_nonzero(planning_backups_post == 16)

    fig, ax = plt.subplots()

    x = ["before shock","after shock"]
    y = [ count_pre , count_post ]
    ax.bar(x,y, width= 0.3, color="orange")
    plt.show()

    return


plot_fig5g()