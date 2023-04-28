from prio_replay.maze import LinearTrack, OpenField
from prio_replay.parameters import Parameters
import numpy as np
from prio_replay.simulation import run_simulation
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
    p.sigma = 0.1
    p.start_rand = False
    p.Tgoal2start = True
    p.onlineVSoffline = "online"



    # variables to store and plot data
    forward = [0] * 50
    backward = [0] * 50
    activation = [0] * 50

    for i in range(nb_sims) :
        print("[fig 6a] : running simulation : "+str(i+1)+"/"+str(nb_sims))

        log = run_simulation(m,p)

        # get the average amount of forward or backward per episode
        forward = [ forward[k] + (log.forward[k]/nb_sims) for k in range(50) ]
        backward = [ backward[k] + (log.backward[k]/nb_sims) for k in range(50) ]   
        activation = [ activation[k] + (log.nb_replay_per_ep[k]/nb_sims) for k in range(50)]

    # calculate the average between forward and backward
    avg_f_b = [ (f + b) / 2 for f, b in zip(forward, backward) ]

    print(activation)
    figure, axis = plt.subplots(2, 2)

    # plot : FORWARD EVENTS PER EPISODE
    axis[0][0].plot(forward)
    axis[0][0].set_title("forward events per episode")

    # plot : BACKWARD EVENTS PER EPISODE
    axis[0][1].plot(backward)
    axis[0][1].set_title("backward events per episode")

    # plot : REPLAY EVENTS (AVERAGE BETWEEN BACKWARD AND FORWARD) PER EPISODE
    axis[1][0].plot(avg_f_b)
    axis[1][0].set_title("average Backward or Forward per episode")

    axis[1][1].plot(activation)
    axis[1][1].set_title("average activation probability per episode")


    plt.show()

    return

plot_fig6a(25)
