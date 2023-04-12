import numpy as np
from maze import Tmaze
from parameters import Parameters
from simulation import run_simulation
import matplotlib.pyplot as plt
from logger import Logger


def plot_4h(nb_sims : int = 500) :
    
    m = Tmaze()
    p = Parameters()

    p.start_rand = False; 


    # Overwrite parameters
    p.MAX_N_STEPS       = int(1e5) # maximum number of steps to simulate
    p.MAX_N_EPISODES    = 50; # maximum number of episodes to simulate (use Inf if no max) -> Choose between 20 and 100
    p.Nplan             = 20; # number of steps to do in planning (set to zero if no planning or to Inf to plan for as long as it is worth it)
    p.onlineVSoffline   = 'offline'; # OFFLINE == AGENT IS ASLEEP
    p.alpha             = 1.0; # learning rate for real experience (non-bayesian)
    p.gamma             = 0.90; # discount factor
    p.tau               = 0.2

   

    # Run simulation 

    # n = [None] * p.MAX_N_STEPS
    # log = [{'replay_state': n, 'replay_action': n} for i in range(nb_sims)]
    
    maze_length = len(m.walls) + m.nb_states
    replayCount = np.zeros((nb_sims, m.nb_states, 4))

    log = Logger()

    for k in range(nb_sims) :
        print("sim" + str(k))
        log = run_simulation(m, p)

        state = log.replay_state[k]
        action = log.replay_action[k]
        # print(state)
        # print(action)

        # Reshape the state and action arrays to a column 
        state_col = np.reshape(state, (-1, 1))
        action_col = np.reshape(action, (-1, 1))
        # print("ACTION")
        # print(action_col)
        # print("STATE")
        # print(state_col)


        # Concatenate the state and action arrays horizontally
        saReplay = np.hstack((state_col, action_col))

        # print(saReplay)
        # print(saReplay[:,0])


        for st in range(m.nb_states):
            for at in range(4):
                # print("BRIDGE: st" + str(st) + ",at" + str(at))
                replayCount[k,st,at] = np.sum((saReplay[:,0]==st) & (saReplay[:,1]==at))
                # print(replayCount[k,st,at])

            
    replayCount = np.mean(replayCount, axis=0)
    # print(replayCount)
    # print(replayCount[2, 2])
    # print(replayCount[4, 2])
    # print(replayCount[2, 3])
    # print(replayCount[1, 3])
    # print(np.sum(replayCount))


    replayRight = replayCount[2,2] + replayCount[4,2]
    replayLeft = replayCount[2, 3] + replayCount[1,3]

    left = replayLeft / np.sum(replayCount)
    right = replayRight / np.sum(replayCount)

    plt.bar(range(2), np.array([left, right]) * 100)
    plt.xticks(range(2), ['UNCUED', 'CUED'])
    plt.ylim([0, 100])

    plt.show()


plot_4h()


