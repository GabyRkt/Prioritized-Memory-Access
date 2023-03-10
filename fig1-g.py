
from maze import LinearTrack, OpenField
from parameters import Parameters
from simulation import run_simulation
from transition_handler import run_pre_explore
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import pandas as pd



import warnings
warnings.filterwarnings("ignore")

def plot_1g(mazetype) :
    if mazetype == "OpenField" :
        m = OpenField()
        current_state = 2 # state where the agent is in figure 1.g
    elif mazetype == "LinearTrack" :
        m = LinearTrack()
        current_state = 4 # state where the agent is in figure 1.g

    p = Parameters()
    
    run_pre_explore(m)
    SR1 = np.linalg.inv(np.eye(len(m.T)) - p.gamma * m.T) # (I - gammaT)^np.NaN

    run_simulation(m,p)
    SR2 = np.linalg.inv(np.eye(len(m.T)) - p.gamma * m.T) # (I - gammaT)^np.NaN
    
    a = [ i/max(SR1[current_state]) for i in SR1[current_state] ]
    b = [ i/max(SR2[current_state]) for i in SR2[current_state] ]

    if mazetype == "OpenField" :
        
        need_b4 = [
            [ a[0], a[6] , a[12] , a[15], a[21], a[27] , a[32], np.NaN, a[41] ],
            [ a[1], a[7] , np.NaN, a[16], a[22], a[28] , a[33], np.NaN, a[42] ],
            [ a[2], a[8] , np.NaN, a[17], a[23], a[29] , a[34], np.NaN, a[43] ],
            [ a[3], a[9] , np.NaN, a[18], a[24], a[30] , a[35], a[38] , a[44] ],
            [ a[4], a[10], a[13] , a[19], a[25], np.NaN, a[36], a[39] , a[45] ],
            [ a[5], a[11], a[14] , a[20], a[26], a[31] , a[37], a[40] , a[46] ]
        ]

        need_after = [
            [ b[0], b[6] , b[12] , b[15], b[21], b[27] , b[32], np.NaN, b[41] ],
            [ b[1], b[7] , np.NaN, b[16], b[22], b[28] , b[33], np.NaN, b[42] ],
            [ b[2], b[8] , np.NaN, b[17], b[23], b[29] , b[34], np.NaN, b[43] ],
            [ b[3], b[9] , np.NaN, b[18], b[24], b[30] , b[35], b[38] , b[44] ],
            [ b[4], b[10], b[13] , b[19], b[25], np.NaN, b[36], b[39] , b[45] ],
            [ b[5], b[11], b[14] , b[20], b[26], b[31] , b[37], b[40] , b[46] ]
        ]
    
    if mazetype == "LinearTrack" :
        need_b4 = [
            [ a[0]  , a[2]  , a[4]  , a[6]  , a[8]  , a[10] , a[12] , a[14] , a[16] , a[18]  ],
            [ np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN ],
            [ a[1]  , a[3]  , a[5]  , a[7]  , a[9]  , a[11] , a[13] , a[15] , a[17] , a[19]  ],
        ]
        need_after = [
            [ b[0]  , b[2]  , b[4]  , b[6]  , b[8]  , b[10] , b[12] , b[14] , b[16] , b[18]  ],
            [ np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN ],
            [ b[1]  , b[3]  , b[5]  , b[7]  , b[9]  , b[11] , b[13] , b[15] , b[17] , b[19]  ],
        ]
    
    sns.set(font_scale=3)
    f1 = plt.figure(figsize=(20,12))
    f1.suptitle(mazetype+': Need Before Learning')

    ax1 = sns.heatmap(need_b4, vmin=0, square=True, yticklabels=False, xticklabels=False, cmap="Blues", linewidths=0.003, linecolor="black")
    ax1.set_facecolor('xkcd:black')
    plt.savefig("figures/fig_1g/"+mazetype+"_before_learning.png")

    f2 = plt.figure(figsize=(20,12))
    f2.suptitle(mazetype+': Need After Learning')

    ax2 = sns.heatmap(need_after, vmin=0,square=True, yticklabels=False, xticklabels=False, cmap="Blues", linewidths=0.003, linecolor="black")
    ax2.set_facecolor('xkcd:black')
    plt.savefig("figures/fig_1g/"+mazetype+"_after_learning.png")


plot_1g("OpenField")

