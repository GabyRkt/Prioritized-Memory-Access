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
    p.tau = 50
    p.epsilon = 0
    p.sigma = 0.1
    p.start_rand = False
    p.Tgoal2start = True
    p.onlineVSoffline = "online"



    # variables to store and plot data
    activation = [0] * 2000


    for i in range(nb_sims) :
        print("[fig 6b] : running simulation : "+str(i+1)+"/"+str(nb_sims))

        log = run_simulation(m,p)

        # get the average amount of forward or backward per episode
        visits = log.steps_per_episode

        for elem in visits :
            print(len(elem))

        for ep_i in range(p.MAX_N_EPISODES) :
            ep_i_visit = visits[ep_i]
            ep_i_activation = log.forward_per_state[ep_i] + log.backward_per_state[ep_i]

            for state in ep_i_activation :
                ind = ep_i_visit.count(state)
                activation[ind] += 1

            
    figure, axis = plt.subplots(1, 1)

    # plot : FORWARD EVENTS PER EPISODE
    axis.plot(activation[0:20])
    axis.set_title("forward/backward replays per number of visits")

    plt.show()

    return

plot_fig6b(50)
