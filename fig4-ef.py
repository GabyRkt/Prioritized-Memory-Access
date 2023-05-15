from prio_replay.maze import LinearTrack, OpenField
from prio_replay.parameters import Parameters
from prio_replay.simulation import run_simulation
from prio_replay.logger import Logger

import matplotlib.pyplot as plt
import numpy as np



def plot_fig4ef(nbsims: int = 1000) :

    m = OpenField()
    p = Parameters()

    p.Talpha = 0.9
    p.alpha = 1
    p.tau = 0.2
    p.gamma = 0.9

    for i in range(nbsims) :
        print("[fig 4.e,f] running simulation : "+str(i+1)+"/"+str(nbsims))

        log = run_simulation(m,p)
        
    








    return

plot_fig4ef(1)