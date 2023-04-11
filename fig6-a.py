from maze import LinearTrack, OpenField
from parameters import Parameters
import numpy as np
from simulation import run_simulation
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def plot_fig6a( nb_sims : int = 100 ) :

    # initializing parameters
    m = LinearTrack()
    p = Parameters()
    p.actpolicy = "softmax"
    p.tau = 0.2
    p.epsilon = 0.1
    p.sigma = 0.5
    p.start_rand = False
    p.Tgoal2start = True
    p.onlineVSoffline = "online"



    # variables to store and plot data
    forward = [0] * 50
    backward = [0] * 50

    for i in range(nb_sims) :
        print("[fig 6a] : running simulation : "+str(i+1)+"/"+str(nb_sims))

        log = run_simulation(m,p)

        # get the average amount of forward or backward per episode
        forward = [ forward[k] + (log.forward[k]/nb_sims) for k in range(50) ]
        backward = [ backward[k] + (log.backward[k]/nb_sims) for k in range(50) ]
    

    # calculate the average between forward and backward
    avg_f_b = [ (f + b) / 2 for f, b in zip(forward, backward) ]



    figure, axis = plt.subplots(1, 3)

    # plot : FORWARD EVENTS PER EPISODE
    axis[0].plot(forward)
    axis[0].set_title("forward events per episode")

    # plot : BACKWARD EVENTS PER EPISODE
    axis[1].plot(backward)
    axis[1].set_title("backward events per episode")

    # plot : REPLAY EVENTS (AVERAGE BETWEEN BACKWARD AND FORWARD) PER EPISODE
    axis[2].plot(avg_f_b)
    axis[2].set_title("replay events per episode")


    plt.show()

    return

plot_fig6a(50)
