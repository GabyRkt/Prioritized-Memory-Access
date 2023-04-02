import numpy as np
import matplotlib.pyplot as plt


def plot_5a():

    softmaxInvT = 3
    
    gain = np.empty(3)

    Qpre = np.array([1,0])
    pA_pre = np.exp(softmaxInvT * Qpre) / np.sum(np.exp(softmaxInvT * Qpre))



    # ----------- Figure a ------------- #

    Qpost = np.array([[-1, 0], [0.5, 0], [4,0]])

    for i in range(len(Qpost)):
        pA_post = np.exp(softmaxInvT * Qpost[i,:]) / np.sum(np.exp(softmaxInvT * Qpost[i,:]))
        EVpre = np.sum(pA_pre * Qpost[i,:])
        EVpost = np.sum(pA_post * Qpost[i,:])
        gain[i] = EVpost - EVpre


    # Plot
    fig, ax = plt.subplots(1,2)

    ax[0].bar(np.arange(len(Qpost)), gain, color='black', width=0.5 )

    ax[0].set_ylim([-0.05, 3])
    ax[0].set_frame_on(False)
    ax[0].axhline(y=0, color='black', linewidth=0.5)


    ax[0].set_xticks(np.arange(len(Qpost)), np.array([-1, 0.5, 3]))
    ax[0].set_yticks([])

    ax[0].set_ylabel('Gain Term')
    ax[0].set_xlabel('Qnew')
    ax[0].set_title('[1,0] -> [Qnew,0]')



    # ----------- Figure b ------------- #

    Qpost = np.array([[1, -1], [1, 0.5], [1, 4]])

    for i in range(len(Qpost)):
        pA_post = np.exp(softmaxInvT * Qpost[i,:]) / np.sum(np.exp(softmaxInvT * Qpost[i,:]))
        EVpre = np.sum(pA_pre * Qpost[i,:])
        EVpost = np.sum(pA_post * Qpost[i,:])
        gain[i] = EVpost - EVpre


    # Plot
    ax[1].bar(np.arange(len(Qpost)), gain, color='black', width=0.5 )

    ax[1].set_ylim([-0.05, 3])
    ax[1].set_frame_on(False)
    ax[1].axhline(y=0, color='black', linewidth=0.5)

    
    ax[1].set_xticks(np.arange(len(Qpost)), np.array([-2, 0.5, 2]))
    ax[1].set_yticks([])

    ax[1].set_xlabel('Qnew')
    ax[1].set_title('[1,0] -> [1,Qnew]')


    plt.show()
    # print(gain)





plot_5a()


