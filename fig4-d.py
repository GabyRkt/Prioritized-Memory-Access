from prio_replay.maze import LinearTrack, OpenField
from prio_replay.parameters import Parameters
from prio_replay.simulation import run_simulation

from prio_replay.distance import breadth_first_search

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def plot_fig4d( nb_sims : int = 1000 ) : 

    m = OpenField()
    p = Parameters()
    p.actpolicy = "softmax"
    p.tau = 0.2
    p.sigma = 0.2
    p.gamma = 0.9
    p.start_rand = True

    dist_rew = []
    dist_agent = []


    for i in range( nb_sims ) :
        print("[fig 4.a] : running simulation : "+str(i+1)+"/"+str(nb_sims))
        log = run_simulation(m,p)

        for (st, plan_st) in log.dist_agent_replay_state :
            
            dist_agent.append( breadth_first_search(plan_st,st) )
            dist_rew.append( breadth_first_search(plan_st,41) )



    # calculating the probability of distance from each state to reward / agent
    rand_dist_rew = []
    rand_dist_agent = []
    for st in range(m.nb_states) :
        rand_dist_rew.append( breadth_first_search(st,41) )
        
        # dist from agent when agent is reaches reward ( state 42 )
        rand_dist_agent.append( breadth_first_search(st,42) )

        # dist from agent when agent starts a run ( random )
        for st2 in range(m.nb_states) :
            rand_dist_agent.append( breadth_first_search(st2,st) )

    prob_rew = [0] * 16
    prob_agent = [0] * 16
    for i in range(16) :
        prob_rew[i] = rand_dist_rew.count(i) 
        prob_agent[i] = rand_dist_agent.count(i) 
    
    prob_rew = [ prob_rew[i]/sum(prob_rew) for i in range(len(prob_rew)) ]
    prob_agent = [ prob_agent[i]/sum(prob_agent) for i in range(len(prob_agent)) ]



    fig, ax = plt.subplots(1,2)
    
    # distance to agent
    ax[0].hist(dist_agent, density=True, range=(0,15), rwidth= 0.6, color='black', bins=15)
    ax[0].plot(prob_agent,linestyle='dashed',color="grey")

    ax[0].set_title("distance to agent")
    ax[0].set_xticks([0,15])
    
    ax[0].hlines(y=[0.1,0.2,0.3,0.4,0.5],xmin=-1, xmax=16,colors='grey',linewidths=0.3)
    ax[0].set_ylim([0, 0.5])

    # distance to reward
    ax[1].hist(dist_rew, density=True, range=(0,15),rwidth= 0.6, color='black',bins=15)
    ax[1].plot(prob_rew,linestyle='dashed',color="grey")

    ax[1].set_title("distance to reward")
    ax[1].hlines(y=[0.1,0.2,0.3,0.4,0.5], xmin=-1, xmax=16, colors='grey',linewidths=0.3)

    ax[1].set_xticks([0, 15])
    ax[1].set_ylim([0, 0.5])
    
    plt.show()

    return



plot_fig4d(10)



