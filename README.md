# Prioritized-Memory-Access

![alt text](https://github.com/GabyRkt/Prioritized-Memory-Access/blob/main/pma.png?raw=true)

Ce projet vise à reproduire les résultats d'un article de neurosciences computationnelles en utilisant des méthodes d'apprentissage par renforcement. Nous avons alors étudié un article réalisé par Marcelo G. Mattar et Nathaniel D. Daw: [Prioritized memory access explains planning and hippocampal replay](https://www.biorxiv.org/content/biorxiv/early/2018/05/20/225664.full.pdf). Il explique comment un rat se sert de sa mémoire pour apprendre à naviguer et à planifier ses déplacements dans un labyrinthe.

Afin d’en apprendre plus sur les méthodes d’apprentissage par renforcement, nous avons effectué des tp sur [la programmation dynamique](https://colab.research.google.com/drive/1ya63BRxbtrkd2RqHqXeSjDKMh5rPGTAi) et [l’apprentissage par renforcement](https://colab.research.google.com/drive/1-p9yoADgETBNY9VqjjLFaI13GHBHPftp). 

De plus, pour réaliser ce projet, nous nous sommes inspirés de nombreux codes notamment [celui des auteurs réalisé en matlab](https://github.com/marcelomattar/PrioritizedReplay), [un code réalisé par des étudiants en Master 2](https://github.com/osigaud/Prioritized-Memory-Access) et [une bibliothèque créée par monsieur Olivier Sigaud](https://github.com/osigaud/SimpleMazeMDP).

Vous trouverez une première version de notre code : [Prioritized Memory Access](https://colab.research.google.com/drive/1gJQQxFmRuTYHx_oU7vF4k3s7POdKIe6T)



# FIGURES CHECKLIST

FIGURE 1 : 
- Number of steps per episode of Prioritized, Random and No Replay (1.d) ✔️
- Link between gain and changes in Q values (1.f) ✔️
- Need values with a random policy and a learned policy (1.g) ✔️

FIGURE 2 :
- Illustrations of Forward and Reverse Replay on the Linear Track ✔️
- Illustrations of Forward and Reverse Replay on the Open Maze ✔️
- Visualisation of the need term when the agent is at the start and the end of the Linear Track ✔️
- Visualisation of the need term when the agent is at the start and the end of the Open Maze ✔️

FIGURE 3 :
- Rate of Forward vs Reverse Replay before and after a run (3) ✔️

FIGURE 4 :
- Distance between agent and replay start (4.a) ✔️
- Activation probability across all backups within an episode (4.c) 
- Probability that a backup happens at various distance from the agent or reward (4.d) 
- How forward replay predict future steps (4.e)
- How reverse replay predict past steps (4.f)
- Proportion of backups leading to the cued vs uncued arm (4.h) ✔️

FIGURE 5 :
- Gain term when the agent learns a new value for the best action (5.a) ✔️
- Gain term when the agent learns a new value for the worst action (5.b) ✔️
- Difference in Forward and Reverse Replay after a reward increase (5.c) ✔️
- Difference in Forward and Reverse Replay after a reward drop to 0 (5.e) ✔️
- Activation probability of aversive zone before and after shock delivery (5.g) ✔️

FIGURE 6 :
- Activation probability over 50 episodes (6.a) ✔️
- Activation probability of a specific state as a function of number of visits per episode (6.b) 










