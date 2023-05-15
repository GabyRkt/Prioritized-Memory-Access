
from prio_replay.maze import LinearTrack, OpenField
from prio_replay.parameters import Parameters
import numpy as np
from prio_replay.simulation import run_simulation
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def plot_fig1d(mazetype : str = "OpenField", fig2plot : list = ["pr","rr","nr"], nb_sims : int = 1000) :
    """ mazetype : "OpenField" or "LinearTrack"
        fig2plot : list containing the names of figures to plot ("pr", "rr", "nr")
        nb_sims : number of simulations, default is set at 1000
    """

    # NO REPLAY PARAMETERS
    p_nr = Parameters()
    p_nr.actpolicy = "egreedy"
    p_nr.epsilon = 0
    p_nr.Nplan = 0

    no_replay = [0] * 50

    # RANDOM REPLAY PARAMETERS
    p_rr = Parameters()
    p_rr.actpolicy = "egreedy"
    p_rr.epsilon = 0
    p_rr.Nplan = 20
    p_rr.allgain2one = True
    p_rr.allneed2one = True

    random_replay = [0] * 50

    # RANDOM REPLAY PARAMETERS
    p_pr = Parameters()
    p_pr.actpolicy = "egreedy"
    p_pr.epsilon = 0
    p_pr.Nplan = 20
    p_pr.allgain2one = False
    p_pr.allneed2one = False

    prio_replay = [0] * 50

    if mazetype == "OpenField" :
        m_nr = OpenField()  # No Replay Maze
        p_nr.start_rand = True

        m_rr = OpenField()  # Random Replay Maze
        p_rr.start_rand = True

        m_pr = OpenField()  # Prioritized 
        p_pr.start_rand = True


    elif mazetype == "LinearTrack" :
        m_nr = LinearTrack() # No Replay Maze
        p_nr.start_rand = False

        m_rr = LinearTrack() # Random Replay Maze
        p_rr.start_rand = False

        m_pr = LinearTrack() # Prioritized 
        p_pr.start_rand = False
    

    # RUN X SIMULATIONS TO GET THE AVERAGE NB STEPS FOR EACH EPISODE
    for i in range(nb_sims) :

        # NO REPLAY SIMULATIONS
        if "nr" in fig2plot : 
            print("\nsimulation NO REPLAY "+ str(i)) 
            log_nr = run_simulation(m_nr,p_nr)
            no_replay = [ no_replay[j] + (log_nr.nbStep[j]/nb_sims) for j in range( len(log_nr.nbStep) ) ]

        # RANDOM REPLAY SIMULATIONS
        if "rr" in fig2plot :
            print("\nsimulation RANDOM REPLAY "+ str(i)) 
            log_rr = run_simulation(m_rr,p_rr)
            random_replay = [ random_replay[j] + (log_rr.nbStep[j]/nb_sims) for j in range( len(log_rr.nbStep) ) ]

        # PRIORITIZED REPLAY SIMULATIONS
        if "pr" in fig2plot :
            print("\nsimulation PRIORITIZED REPLAY "+ str(i)) 
            log_pr = run_simulation(m_pr,p_pr)
            prio_replay = [ prio_replay[j] + (log_pr.nbStep[j]/nb_sims) for j in range( len(log_pr.nbStep) ) ]
    

    # PLOT FIGURES
    xx = [i for i in range(21)]

    if "nr" in fig2plot :
        plt.plot(xx,no_replay[0:21],"r-",label = "no replay")
    if "rr" in fig2plot :
        plt.plot(xx,random_replay[0:21],"b-", label = "random replay")
    if "pr" in fig2plot :
        plt.plot(xx,prio_replay[0:21],"g-", label = "prioritized replay")

    plt.title( mazetype + " | Steps per episode")
    plt.xticks(range(21), xx)
    plt.legend()
    plt.ylim(0,200)
    plt.show()


plot_fig1d("LinearTrack",["nr","rr","pr"],500)








