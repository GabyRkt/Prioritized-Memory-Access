from maze import Tmaze
from parameters import Parameters
from simulation import run_simulation

def plot_4h(nb_sims : int = 1000) :
    
    p = Parameters

    p.start_rand = False; 

    # Overwrite parameters


    p.MAX_N_STEPS       = 1e5
    p.MAX_N_EPISODES    = 50
    p.MAX_N_EPISODES    = 50; # maximum number of episodes to simulate (use Inf if no max) -> Choose between 20 and 100
    p.Nplan             = 20; # number of steps to do in planning (set to zero if no planning or to Inf to plan for as long as it is worth it)
    p.onlineVSoffline   = 'off-policy'; # Choose 'off-policy' (default, learns Q*) or 'on-policy' (learns Qpi) learning for updating Q-values and computing gain
    p.alpha             = 1.0; # learning rate for real experience (non-bayesian)
    p.gamma             = 0.90; # discount factor
    p.tau               = 0.2


    # Run simulation 

    # for k in range(nb_sims) :
        


