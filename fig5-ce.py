from maze import LinearTrack, OpenField
from parameters import Parameters
import numpy as np
from simulation import run_simulation
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def plot_fig5ce( nb_sims : int = 100 ) :

    m = LinearTrack()
    p = Parameters()
    p.actpolicy = "softmax"
    p.tau = 0.2
    p.epsilon = 0.1
    p.start_rand = False
    p.Tgoal2start = True

    # variables to plot
    x1_forward = 1        # number of forward events when r isnt changed (r = 1)
    x1_backward = 1       # number of backward events when r isnt changed (r = 1)

    x4_forward = 1        # number of forwards after an increase in r (r = 4)
    x4_backward = 1       # number of backwards after an increase in r (r = 4)

    x0_forward = 1        # number of forwards after a drop in r (r = 0)
    x0_backward = 1      # number of forwards after a drop in r (r = 0)


    # finding the number of forwards & backwards when r = 4 every other for 25/50 episodes
    p.changeR = True ; p.x4 = True ; p.x0 = False

    for i in range( nb_sims ) :
        print("[fig 5c] : running simulation : "+str(i+1)+"/"+str(nb_sims))
        log = run_simulation(m,p)

        x1_forward += sum( [ log.forward[k1] for k1 in range(50) if k1 not in [2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,34,35,38,39,42,43,46,47] ] )
        x4_forward += sum( [ log.forward[k1] for k1 in [2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,34,35,38,39,42,43,46,47] ] )

        x1_backward += sum( [ log.backward[k1] for k1 in range(50) if k1 not in [2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,34,35,38,39,42,43,46,47] ] )
        x4_backward += sum( [ log.backward[k1] for k1 in [2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,34,35,38,39,42,43,46,47] ] )

    x = [ "1x/1x" , "4x/1x" ]

    y_forward = [ x1_forward - x1_forward , (x4_forward - x1_forward)/x1_forward]
    y_backward = [ x1_backward - x1_backward , (x4_backward - x1_backward)/x1_backward]

    figure, axis = plt.subplots(2, 2)

    # fig 5 c : forward
    axis[0,0].bar(x,y_forward)
    axis[0,0].set_title("Forward after increase in r (1->4)")
    axis[0,0].axhline(y=0,linewidth=1, color='k')
    axis[0,0].set_ylim([-5,5])

    # fig 5 c : backward
    axis[0,1].bar(x,y_backward)
    axis[0,1].set_title("Backward after increase in r (1->4)")
    axis[0,1].axhline(y=0,linewidth=1, color='k')
    axis[0,1].set_ylim([-5,5])

    # finding the number of forwards & backwards when r = 0 every other for 25/50 episodes
    p.changeR = True ; p.x4 = False ; p.x0 = True

    for i in range( nb_sims ) :
        print("[fig 5e] : running simulation (2/2) : "+str(i+1)+"/"+str(nb_sims))
        log = run_simulation(m,p)

        x1_forward += sum( [ log.forward[k1] for k1 in range(50) if k1 not in [2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,34,35,38,39,42,43,46,47] ] )
        x0_forward += sum( [ log.forward[k1] for k1 in [2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,34,35,38,39,42,43,46,47] ] )

        x1_backward += sum( [ log.backward[k1] for k1 in range(50) if k1 not in [2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,34,35,38,39,42,43,46,47] ] )
        x0_backward += sum( [ log.backward[k1] for k1 in [2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,34,35,38,39,42,43,46,47] ] )

    x = [ "1x/1x" , "0x/1x" ]

    y_forward = [ x1_forward - x1_forward , (x0_forward - x1_forward)/x1_forward]
    y_backward = [ x1_backward - x1_backward , (x0_backward - x1_backward)/x1_backward]

    # fig 5 e : forward
    axis[1,0].bar(x,y_forward)
    axis[1,0].set_title("Forward after drop in r (1->0)")
    axis[1,0].axhline(y=0,linewidth=1, color='k')
    axis[1,0].set_ylim([-5,5])

    # fig 5 e : backward
    axis[1,1].bar(x,y_backward)
    axis[1,1].set_title("Forward after drop in r (1->0)")
    axis[1,1].axhline(y=0,linewidth=1, color='k')
    axis[1,1].set_ylim([-5,5])

    plt.show()

    return

plot_fig5ce(1)