from prio_replay.maze import LinearTrack, OpenField
from prio_replay.parameters import Parameters
import numpy as np
from prio_replay.simulation import run_simulation
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def plot_fig6b( nb_sims : int = 100 ) :

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
    activation = [0] * 50

    for i in range(nb_sims) :
        print("[fig 6b] : running simulation : "+str(i+1)+"/"+str(nb_sims))

        log = run_simulation(m,p)

        # get the average amount of forward or backward per episode
        forward = [ forward[k] + (log.forward[k]/nb_sims) for k in range(50) ]
        backward = [ backward[k] + (log.backward[k]/nb_sims) for k in range(50) ]
        activation = [ activation[k] + (sum(log.nb_backups_per_state[k])/nb_sims) for k in range(50)]

    # calculate the average between forward and backward
    avg_f_b = [ (f + b) / 2 for f, b in zip(forward, backward) ]


    figure, axis = plt.subplots(1, 1)

    # plot : FORWARD EVENTS PER EPISODE
    axis.plot(forward)
    axis.set_title("forward events per episode")

    plt.show(10)

    return

plot_fig6b(1)
