from prio_replay.parameters import Parameters

class Logger :
  """
  This class contains results found during a simulation.
  """

  def __init__(self) :

    "stored data"
    # data for fig 1.d
    self.nbStep = []


    # data for fig 3
    self.nbStep_forwardReplay_forward = 0
    self.nbStep_forwardReplay_backward = 0
    self.nbStep_backwardReplay_forward = 0
    self.nbStep_backwardReplay_backward = 0

    # data for fig 4a
    self.dist_agent_replay_start = []

    # data for fig 4c
    self.nb_backups_per_state = []

    # data for fig 4d
    self.dist_agent_replay_state = []

    # data for fig 4ef 
    self.steps_per_episode = []
    self.backward_per_state = []
    self.forward_per_state = []

    # data for fig 4h
    p = Parameters()
    self.replay_state = [None] * p.MAX_N_STEPS

    self.replay_action = [None] * p.MAX_N_STEPS


    # data for fig 5c and fig 5e
    self.event_episode = []
    self.forward = [0 for _ in range(50)]
    self.backward= [0 for _ in range(50)]

    # data for fig 6a
    self.nb_replay_per_ep = [0 for _ in range(50)]

    # data for fig 6b
    self.nbvisits_per_state = []

  

