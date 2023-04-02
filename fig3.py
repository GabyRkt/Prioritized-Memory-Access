from maze import LinearTrack, OpenField
from parameters import Parameters
from simulation import run_simulation
import matplotlib.pyplot as plt
import numpy as np


def plot_fig3(mazetype: str="LinearTrack", nb_sims: int = 100) :

    if mazetype == "LinearTrack" :
        m = LinearTrack()
        p = Parameters()
        p.start_rand = False
        p.Tgoal2start = True
    
    else :
        m = OpenField()
        p = Parameters()
    
    p.epsilon = 0.1
    p.Talpha = 0.9
    p.tau = 0.2
    p.alpha = 1
    p.gamma = 0.9

    before_run = [ 0 , 0 ]
    after_run = [ 0 , 0 ]

    for i in range(nb_sims) :
        print("running simulation : "+str(i)+"/"+str(nb_sims) )
        log = run_simulation(m,p)

        before_run[0] += log.nbStep_forwardReplay_forward
        before_run[1] += log.nbStep_backwardReplay_forward

        after_run[0] += log.nbStep_forwardReplay_backward
        after_run[1] += log.nbStep_backwardReplay_backward

    total_events = before_run[0] + before_run[1] + after_run[0] + after_run[1]
    before_run = [ a / total_events for a in before_run ]
    after_run = [ b / total_events for b in after_run ] 

    x = np.arange(2)

    print(before_run)
    print(after_run)

    plt.bar(x-0.2, before_run, align='center', width=0.2, color="lightgrey")
    plt.bar(x+0.2, after_run, align='center', width=0.2, color = "black")
    plt.xticks(x, ["Forward Replay","Backward Replay"])
    plt.legend(["Before run", "After run"])
    plt.show()

   

    return

plot_fig3("LinearTrack",10)


