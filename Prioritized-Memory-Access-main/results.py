

class Resulats :
  """
  This class contains results found during a simulation.
  """

  def __init__(self, 
               figure_1_d=False, 
               figure_1_g=False,
               figure_1_f=False,
               
               figure_2_a=False,
               figure_2_b=False,
               figure_2_d=False,
               figure_2_e=False,
               figure_2_g=False,
               figure_2_h=False,
               figure_2_j=False,
               figure_2_k=False,
               
               figure_3=False,

               figure_4_a=False,
               figure_4_c=False,
               figure_4_d=False,
               figure_4_e=False,
               figure_4_f=False,
               figure_4_h=False,

               figure_5_a=False,
               figure_5_b=False,
               figure_5_c=False,
               figure_5_e=False,
               figure_5_g=False,

               figure_6_a=False,
               figure_6_c=False,
               ) :
    
    "depending on which figure we want to print, we store differents data"
    self.figure_1_d = figure_1_d
    self.figure_1_g = figure_1_g     
    self.figure_1_f = figure_1_f
    
    self.figure_2_a=figure_2_a
    self.figure_2_b=figure_2_b
    self.figure_2_d=figure_2_d
    self.figure_2_e=figure_2_e
    self.figure_2_g=figure_2_g
    self.figure_2_h=figure_2_h
    self.figure_2_j=figure_2_j
    self.figure_2_k=figure_2_k
    
    self.figure_3=figure_3

    self.figure_4_a=figure_4_a
    self.figure_4_c=figure_4_c
    self.figure_4_d=figure_4_d
    self.figure_4_e=figure_4_e
    self.figure_4_f=figure_4_f
    self.figure_4_h=figure_4_h

    self.figure_5_a=figure_5_a
    self.figure_5_b=figure_5_b
    self.figure_5_c=figure_5_c
    self.figure_5_e=figure_5_e
    self.figure_5_g=figure_5_g

    self.figure_6_a=figure_6_a
    self.figure_6_c=figure_6_c


    "stored data"
    self.nbStep = -1
    self.need = [] 
    self.gain = []
    self.Q = []
    self.forward_before_run = -1
    self.forward_after_run = -1
    self.backward_before_run = -1
    self.backward_after_run = -1
    self.nbStep_forwardReplay_forward = -1
    self.nbStep_forwardReplay_backward = -1
    self.nbStep_backwardReplay_forward = -1
    self.nbStep_backwardReplay_backward = -1
    self.dist_agent_replay_state = []

    init_variable(self)

    def init_variable(self) :
      if self.figure_1_d :
        ...
    
