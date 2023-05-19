from prio_replay.maze import LinearTrack, OpenField
from prio_replay.parameters import Parameters
from prio_replay.simulation import run_simulation
from prio_replay.logger import Logger

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def plot_fig4ef(nbsims: int = 1000) :

    m = OpenField()
    p = Parameters()

    p.Talpha = 0.9
    p.alpha = 1
    p.tau = 0.2
    p.gamma = 0.9

    forward_n_steps = [0] * 21
    backward_n_steps = [0] * 21

    for i in range(nbsims) :
        print("[fig 4.e,f] running simulation : "+str(i+1)+"/"+str(nbsims))

        log = run_simulation(m,p)
        
        f = log.forward_per_state
        b = log.backward_per_state
        step = log.steps_per_episode

        for ep_i in range(p.MAX_N_EPISODES) :
            for k in range(5) :
                for kk in range(10) :
                    
                    if ep_i != 0 :
                        #forward replay : steps in the future
                        if f[ep_i][k] == step[ep_i][kk] :
                            forward_n_steps[kk +10] += 1
                        
                        #forward replay : steps in the past
                        if f[ep_i][k] == step[ep_i -1][-(kk+1)] :
                            forward_n_steps[kk] += 1

                    if ep_i != 49 :
                        #backward replay : steps in the future
                        if b[ep_i][k] == step[ep_i+1][kk] :
                            backward_n_steps[kk + 10] += 1
                        
                        #backward replay : steps in the past
                        if b[ep_i][k] == step[ep_i][-(kk+1)] :
                            backward_n_steps[kk] += 1

    forward_n_steps = [ elem / sum(forward_n_steps) for elem in forward_n_steps ]
    backward_n_steps = [ elem / sum(backward_n_steps) for elem in backward_n_steps ]
    forward_n_steps[10] = np.NaN
    backward_n_steps[10] = np.NaN

    figure, axis = plt.subplots(2)
    x = [i for i in range(-10,11)]

    axis[0].plot(x,forward_n_steps)
    axis[1].plot(x,backward_n_steps)
    
    plt.show()

    return

plot_fig4ef(500)