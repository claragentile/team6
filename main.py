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
    if (P_iter[-1] != P[i]).all():
        answer += 1

if answer != 0:
    print('the first test failed, there are', answer, 'differences')
elif answer == 0:
    print('the first test passed')

patterns = f.generate_patterns(80,1000) 
answer_2 = 0 

for i in range(np.size(patterns,0)):
    perturbed_pattern = f.perturb_pattern(patterns[i], 200)
    historic_pertubed_p = f.dynamics_async(perturbed_pattern, f.hebbian_weights(patterns), 10000,3000)
    if (historic_pertubed_p[-1] != patterns[i]).all():
        answer_2 += 1

if answer_2 != 0:
    print('the second test failed, there are', answer_2, 'differences')
else :
    print('the second test passed')
