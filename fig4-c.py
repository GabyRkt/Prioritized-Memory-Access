from prio_replay.maze import LinearTrack, OpenField
from prio_replay.parameters import Parameters
from prio_replay.simulation import run_simulation

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def plot_fig4c( nbsims : int = 1000 ) : 
    
    m = OpenField()
    p = Parameters()
    p.actpolicy = "softmax"
    p.tau = 0.2
    p.sigma = 0.4
    p.gamma = 0.9
    p.start_rand = True


    a = [0] * m.nb_states
    tmp = [[0] * m.nb_states ] * p.MAX_N_EPISODES

    for i in range(nbsims) :
        print("[fig 4c] : running simulation : "+str(i+1)+"/"+str(nbsims))
        
        log = run_simulation(m,p)
        
        nb_backups_per_episode = [ sum(log.nb_backups_per_state[k]) for k in range(len(log.nb_backups_per_state)) ]
        
        for nb_eps in range(len(log.nb_backups_per_state)) :
            tmp[nb_eps] = [ tmp[nb_eps][st] + (log.nb_backups_per_state[nb_eps][st] / nb_backups_per_episode[nb_eps] / nbsims )  for st in range( m.nb_states ) ] 


    for st in range(m.nb_states) :
        for nb_eps in range(p.MAX_N_EPISODES) :
            a[st] = a[st] + tmp[nb_eps][st]/(p.MAX_N_EPISODES)

    # turn the result into a percentage out of 100
    a = [ elem * 100 for elem in a ]
   
    a = [
            [ a[0], a[6] , a[12] , a[15], a[21], a[27] , a[32], np.NaN, a[41] ],
            [ a[1], a[7] , np.NaN, a[16], a[22], a[28] , a[33], np.NaN, a[42] ],
            [ a[2], a[8] , np.NaN, a[17], a[23], a[29] , a[34], np.NaN, a[43] ],
            [ a[3], a[9] , np.NaN, a[18], a[24], a[30] , a[35], a[38] , a[44] ],
            [ a[4], a[10], a[13] , a[19], a[25], np.NaN, a[36], a[39] , a[45] ],
            [ a[5], a[11], a[14] , a[20], a[26], a[31] , a[37], a[40] , a[46] ]
        ]

    x1 = sns.heatmap(a, vmin=0, annot=True, square=True, yticklabels=False, xticklabels=False, linewidths=0.003, linecolor="black")
    plt.show()


    return


plot_fig4c(20)
