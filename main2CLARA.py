import functions as f
import numpy as np

patterns = f.generate_patterns(80,1000) 

answer = 0 

for i in range(np.size(patterns,0)):
    perturbed_pattern = f.perturb_pattern(patterns[i], 200)
    historic_pertubed_p = f.dynamics_async(perturbed_pattern, f.hebbian_weights(patterns), 20000,3000)
    if (historic_pertubed_p[-1] != patterns[i]).all():
        answer += 1

if answer != 0:
    print('the first test failed, there are', answer, 'differences')
elif answer == 0:
    print('the first test passed')