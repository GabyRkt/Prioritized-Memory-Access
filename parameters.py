
"""
This class contains parameters used for an agent's training .
"""


class Parameters :

  def __init__(self) :
    
    self.MAX_N_STEPS = int(1e5)
    self.MAX_N_EPISODES = 50

    #params for pre-exploration :

    self.preExplore = True  
    self.Tgoal2start = True   

    #params for action selection policies :

    self.actpolicy = "softmax"    # "softmax" or else "egreedy"
    self.epsilon = 0.1    # exploration rate for egreedy
    self.tau = 1/5    # inverse temperature for softmax

    #params for Q-learning :

    self.gamma = 0.9    # discount rate
    self.alpha = 1.0    # learning rate
    self.lmbda = 0.1    # eligibility trace's decay rate

    #params for Transition Matrix :
    
    self.Talpha = 0.9

    #params for planning

    self.planOnlyAtGorS = True
    self.Nplan = 20
    self.remove_samestate = True
    self.expandFurther = True
    self.onlineVSoffline = "online"    # "online" = Successor Representation
    self.allowLoops = False    # allow loops in n-step backups
    self.baselineGain = int(1e10)
    self.EVBthresh = 0




