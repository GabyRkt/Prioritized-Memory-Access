
from maze import LinearTrack, OpenField
from parameters import Parameters
from simulation import run_simulation, evaluate
import matplotlib.pyplot as plt



o = OpenField()
params = Parameters()

o.reInit()

q_list = run_simulation(o,params)
list_steps = evaluate(o,q_list,params)


plt.plot(list_steps)
plt.title("\n OPEN FIELD :\n   steps after each episode   \n")
plt.show()









