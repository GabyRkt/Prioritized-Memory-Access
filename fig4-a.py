from prio_replay.maze import LinearTrack, OpenField
from prio_replay.parameters import Parameters
from prio_replay.simulation import run_simulation

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def plot_fig4a(nb_sims: int = 100) :

    m = LinearTrack()
    p = Parameters()
    p.actpolicy = "softmax"
    p.tau = 0.2
    p.epsilon = 0.1

    dist = []

    for i in range( nb_sims ) :
        print("[fig 4.a] : running simulation : "+str(i+1)+"/"+str(nb_sims))
        log = run_simulation(m,p)

        for (st, plan_st) in log.dist_agent_replay_start :
            
            if st%2 == 0 :
                dist.append( int( (plan_st-st)//2 ) )
            
            else :
                dist.append( int( (st-plan_st)//2) )
    
    x = [i for i in range(-10,11)]
    y = [dist.count(i)/len(dist) for i in range(-10,11) ]

    plt.bar(x, y)
    plt.ylim(top=1)
    plt.show()

    return

plot_fig4a()
