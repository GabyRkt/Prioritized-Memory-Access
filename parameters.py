class Parameters :
  """
  This class contains settings used for an agent's training.
  """

  def __init__(self) :
    
    self.MAX_N_STEPS = 40000       
    "maximum number of steps in a simulation"

    self.MAX_N_EPISODES = 50          
    "maximum number of episodes in a simulation"

    self.start_rand = True
    "is the starting state random or pre-determined"

    #params for pre-exploration :

    self.preExplore = True            
    "lets the agent explore the maze without rewards before the first episode"

    self.Tgoal2start = True           
    "includes a transition from goal to start in transition matrix"


    #params for action selection policies :

    self.actpolicy = "softmax"        
    "softmax or egreedy"

    self.epsilon = 0.1               
    "exploration rate for egreedy"

    self.tau = 0.2                  
    "inverse temperature for softmax"


    #params for Q-learning :

    self.gamma = 0.9
    "discount rate"    
    
    self.alpha = 1.0  
    "learning rate"  
    
    self.lmbda = 0
    "eligibility trace's decay rate"


    #params for Transition Matrix :
    
    self.Talpha = 0.9                 
    "learning rate for the transition matrix"


    #params for planning:

    self.planOnlyAtGorS = True        
    "indicates if the planning should only happen if the agent is at the start or goal state"

    self.Nplan = 20                   
    "number of steps in a planning (planning_backup)"

    self.remove_samestate = True      
    "removes actions that lead to the same state (i.e. hitting a wall)"
    
    self.expandFurther = True         
    "expand the last backup further"

    self.onlineVSoffline = "online"   # "online" = Successor Representation

    self.allowLoops = False        
    "allow loops in n-step backups"

    self.baselineGain = 1e-10     
    "gain is set to at least this value"

    self.EVBthreshold = 0                
    "minimum EVB so that planning is performed"

    # NEW PARAMETERS FOR RANDOM REPLAY
    self.allneed2one = False
    self.allgain2one = False




