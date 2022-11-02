import functions as f
import numpy as np
#print(f.generate_patterns(3,4))
P = f.generate_patterns(80,1000)
P_perturbe = f.perturb_pattern(P[0], 200)
P_iter = f.dynamics(P_perturbe, f.hebbian_weights(P), 20)
"""""
if (P_iter == P[0]).all():
    print("same at the end")
else :
    print("shit")
"""

answer = 0 
for i in range(np.size(P,0)):
    P_perturbe = f.perturb_pattern(P[i], 200)
    P_iter = f.dynamics(P_perturbe, f.hebbian_weights(P), 20)
    if (P_iter != P[i]).all():
        answer += 1

if answer != 0:
    print('the first test failed, there are', answer, 'differences')
elif answer == 0:
    print('the first test passed')
