
from maze import LinearTrack, OpenField
from parameters import Parameters
from simulation import run_simulation
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def plot_fig1d(mazetype : str = "OpenField", fig2plot : list = ["pr","rr","nr"], nb_sims : int = 1000) :
    """ mazetype : "OpenField" or "LinearTrack"
        fig2plot : list containing the names of figures to plot ("pr", "rr", "nr"
        nb_sims : number of simulations, default is set at 1000
    """

    if mazetype == "OpenField" :
        m_nr = OpenField()  # No Replay Maze
        m_rr = OpenField()  # Random Replay Maze
        m_pr = OpenField()  # Prioritized 
    elif mazetype == "LinearTrack" :
        m_nr = LinearTrack()
        m_rr = LinearTrack()
        m_pr = LinearTrack()
    
    # NO REPLAY PARAMETERS
    p_nr = Parameters()
    p_nr.Nplan = 0

    no_replay = [0] * 50

    # RANDOM REPLAY PARAMETERS
    p_rr = Parameters()
    p_rr.Nplan = 20
    p_rr.allgain2one = True
    p_rr.allneed2one = True

    random_replay = [0] * 50

    # RANDOM REPLAY PARAMETERS
    p_pr = Parameters()
    p_pr.Nplan = 20
    p_pr.allgain2one = False
    p_pr.allneed2one = False

    prio_replay = [0] * 50

    # RUN X SIMULATIONS TO GET THE AVERAGE NB STEPS FOR EACH EPISODE
    for i in range(nb_sims) :

        # NO REPLAY SIMULATIONS
        if "nr" in fig2plot : 
            print("\nsimulation NO REPLAY "+ str(i)) 
            sim_i_nr = run_simulation(m_nr,p_nr)
            no_replay = [ no_replay[j] + (sim_i_nr[j]/nb_sims) for j in range( len(sim_i_nr) ) ]

        # RANDOM REPLAY SIMULATIONS
        if "rr" in fig2plot :
            print("\nsimulation RANDOM REPLAY "+ str(i)) 
            sim_i_rr = run_simulation(m_rr,p_rr)
            random_replay = [ random_replay[j] + (sim_i_rr[j]/nb_sims) for j in range( len(sim_i_rr) ) ]

        # PRIORITIZED REPLAY SIMULATIONS
        if "pr" in fig2plot :
            print("\nsimulation PRIORITIZED REPLAY "+ str(i)) 
            sim_i_pr = run_simulation(m_pr,p_pr)
            prio_replay = [ prio_replay[j] + (sim_i_pr[j]/nb_sims) for j in range( len(sim_i_pr) ) ]
    

    # PLOT FIGURES
    if "nr" in fig2plot :
        plt.plot(no_replay[0:21],"r-",label = "no replay")
    if "rr" in fig2plot :
        plt.plot(random_replay[0:21],"b-", label = "random replay")
    if "pr" in fig2plot :
        plt.plot(prio_replay[0:21],"g-", label = "prioritized replay")

    plt.title( mazetype + " | Steps per episode")
    plt.legend()
    plt.ylim(0,200)
    plt.show()


plot_fig1d("OpenField",["nr"],10)







