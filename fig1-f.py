import numpy as np
import matplotlib.pyplot as plt
from parameters import Parameters

def plot_1f():

    params = Parameters

    # Set parameters
    params.actpolicy = 'softmax'

    Q = np.arange(0, 1.001, 0.001)
    Qspacing = 0.1 # Spacing between Q-values before update
    Qrange = 3*Qspacing # Range of Q-vaues to plot (before and after the original Q)

    softmaxInvT = np.array([1, 2, 5, 10, 20, 50])
    params.baselineGain = -np.inf

    grayrange = [0.8, 0.3]
    yRange = [-0.1, 0.2]

    # Initialize variables
    gainOPT = np.empty((len(Q), len(softmaxInvT)))
    gainNOPT = np.empty((len(Q), len(softmaxInvT)))
    #gainNOPT = np.full((len(Q), len(softmaxInvT)), np.nan)


    # Calculate gain for each softmaxInvT
    for i in range(len(softmaxInvT)):

        # Learning about OPTIMAL action

        Qmean = np.array([np.median(Q) + Qspacing, np.median(Q)])

        # Probabilities before update 
        pA_pre = np.exp(softmaxInvT[i] * Qmean) / np.sum(np.exp(softmaxInvT[i] * Qmean))

        # Probabilities after update
        # Stacks 1-D array as columns into a 2-D array
        Qpost = np.column_stack((Q, np.tile(Qmean[1], (len(Q), 1))))

        pA_post = np.exp(softmaxInvT[i] * Qpost) / np.tile(np.sum(np.exp(softmaxInvT[i] * Qpost), axis=1), (2, 1)).T
        

        gainOPT[:, i] = np.sum(Qpost * pA_post, axis=1) - np.dot(Qpost, pA_pre)


        # Learning about NON-OPTIMAL action

        Qmean = np.array([np.median(Q)-Qspacing, np.median(Q)])

        # Probabilities before update
        pA_pre = np.exp(softmaxInvT[i]*Qmean) / np.sum(np.exp(softmaxInvT[i]*Qmean))

        # Probabilities after update
        Qpost = np.column_stack((Q, np.tile(Qmean[1], (len(Q), 1))))

        pA_post = np.exp(softmaxInvT[i]*Qpost) / np.tile(np.sum(np.exp(softmaxInvT[i]*Qpost), axis=1), (2, 1)).T

        gainNOPT[:, i] = np.sum(Qpost*pA_post, axis=1) - np.dot(Qpost, pA_pre)

    

    gainOPT = np.maximum(gainOPT, params.baselineGain)
    gainNOPT = np.maximum(gainNOPT, params.baselineGain)



    # Plot gains 
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    grayvals = grayrange[0] + np.linspace(0, 1, len(softmaxInvT)) * (grayrange[1] - grayrange[0])

    legendvals = []

    for i in reversed(range(len(softmaxInvT))):
 
        if softmaxInvT[i] == np.max(softmaxInvT):
            axs[0].plot(Q, gainOPT[:, i], color = 'b', linewidth = 1.5) 
            axs[1].plot(Q, gainNOPT[:, i], color = 'b', linewidth = 1.5)   
        else:  
            axs[0].plot(Q, gainOPT[:, i], color = np.tile(grayvals[i], 3), linewidth = 1)
            axs[1].plot(Q, gainNOPT[:, i], color = np.tile(grayvals[i], 3), linewidth = 1)
    
        legendvals.append('beta=%.2f' %softmaxInvT[i])
   
    
    # OPTIMAL ACTION
    
    # axs[0].set_xticks(np.arange(np.median(Q) + Qspacing - Qrange, np.median(Q) + Qspacing + Qrange + Qspacing, Qspacing))
    # axs[0].set_yticks(np.arange(-0.1, 0.4, 0.1))
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[0].set_ylabel('Gain')

    axs[0].set_ylim(yRange)
    axs[0].set_xlim([(np.median(Q) + Qspacing - Qrange), (np.median(Q) + Qspacing + Qrange)])
    l11 = axs[0].axvline(np.median(Q), color = 'k', linewidth = 0.5)
    l12 = axs[0].axvline(np.median(Q) + Qspacing, color = 'k', linewidth = 0.5)
    l13 = axs[0].axhline(0, color = 'k', linewidth = 0.5)
    
    axs[0].legend(legendvals, loc = 'upper right')
    axs[0].set_title('Optimal action')


    # NON OPTIMAL ACTION

    # axs[1].set_xticks(np.arange(np.median(Q) - Qspacing - Qrange, np.median(Q) - Qspacing + Qrange + Qspacing, Qspacing))
    # axs[1].set_yticks(np.arange(-0.1, 0.4, 0.1))
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    axs[1].set_ylim(yRange)
    axs[1].set_xlim([(np.median(Q) - Qspacing - Qrange), (np.median(Q) - Qspacing + Qrange)])
    l21 = axs[1].axvline(np.median(Q), color = 'k', linewidth = 0.5)
    l22 = axs[1].axvline(np.median(Q) - Qspacing, color = 'k', linewidth = 0.5)
    l23 = axs[1].axhline(0, color = 'k', linewidth = 0.5)
    
    axs[1].legend(legendvals, loc = 'upper left')
    axs[1].set_title('Non-Optimal action')

    plt.show()

plot_1f()



