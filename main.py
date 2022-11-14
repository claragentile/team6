import main_functions as f
import numpy as np

# test dynamics-----------------------------------------------------------------
P = f.generate_patterns(80,1000)

P_iter = f.dynamics(P[0], f.hebbian_weights(P), 20,200)

if ((P_iter[-1][:] == P[0]).all()):
    print("same at the end")
else :
    print("not same at the end")


answer = 0 
for i in range(np.size(P,0)):
    P_perturbed = f.perturb_pattern(P[i],200)
    P_iter = f.dynamics(P_perturbed, f.hebbian_weights(P),20)
    if (P_iter[-1][:] != P[i]).any():
        answer += 1

if answer != 0:
    print('the test of dynamics with hebbian weights are', answer, 'differences') #the first test didn't failed but some patterns don't converge
elif answer == 0:
    print('the test of dynamics with hebbian weights passed, there are 0 differences')

# test dynamics_async-----------------------------------------------------------------

patterns = f.generate_patterns(80,1000) 

answer_2 = 0 
for i in range(np.size(patterns,0)):
    P_perturbed = f.perturb_pattern(patterns[i], 200)
    historic_pertubed_p = f.dynamics_async(P_perturbed, f.hebbian_weights(patterns), 20000,3000)
    if (historic_pertubed_p[-1][:] != patterns[i]).any():
        answer_2 += 1

if answer_2 != 0:
    print('the test of dynamics_async with hebbian weights are', answer_2, 'differences')
else :
    print('the test of dynamics with hebbian weights passed, there are 0 difference')


#test of storkey dynamics -------------------------------------------------------------------------------------------------------

patterns = f.generate_patterns(80,1000)

answer = 0 
for i in range(np.size(P,0)):
    P_perturbed = f.perturb_pattern(P[i],200)
    P_iter = f.dynamics(P_perturbed,f.storkey_weights(patterns),20)
    if (P_iter[-1][:] != P[i]).any():
        answer += 1

if answer != 0:
    print('the test of dynamics with storkey weights are', answer, 'differences')
elif answer == 0:
    print('the test of dynamics with storkey weights passed, there are 0 differences')

#test of storkey dynamics_async -------------------------------------------------------------------------------------------------------

patterns = f.generate_patterns(80,1000) 
answer_2 = 0 

for i in range(np.size(patterns,0)):
    historic_pertubed_p = f.dynamics_async(patterns[i], f.storkey_weights(patterns), 20000,3000,200)
    if (historic_pertubed_p[-1] != patterns[i]).any():
        answer_2 += 1

if answer_2 != 0:
    print('the second test are', answer_2, 'differences')
else :
    print('the second test passed, there are 0 differences')

#test of energy--------------------------------------------------------------------------------------------------
 
patterns = f.generate_patterns(50, 2500)
position = np.random.randint(0,np.size(patterns,0)) 
perturbed_pattern = f.perturb_pattern(patterns[position], 1000) #pattern choisi pour le test

network_state_list_h_s = f.dynamics(patterns[position], f.hebbian_weights(patterns), 20,1000)
network_state_list_s_s = f.dynamics(perturbed_pattern, f.storkey_weights(patterns), 20,1000)
 
network_state_list_h_a = f.dynamics_async(perturbed_pattern, f.hebbian_weights(patterns), 30000,10000,1000)
network_state_list_s_a = f.dynamics_async(perturbed_pattern, f.storkey_weights(patterns), 30000,10000,1000)
 
print("time-energy plot for the update rule and hebbian weights: ")
f.hebbian_plot_energy_time(network_state_list_h_s)
 
print("time-energy plot for the asynchronous update rule and hebbian weights: ")
f.hebbian_plot_energy_time(network_state_list_h_a)
 
print("time-energy plot for the update rule and storkey weights: ")
f.storkey_plot_energy_time(network_state_list_s_s, 100)
 
print("time-energy plot for the asynchronous update rule and storkey weights: ")
f.storkey_plot_energy_time(network_state_list_s_a, 100)
 
#-test of visualization-------------------------------------------------------------------------------------------------------------
patterns = f.generate_patterns(50, 2500)
board= f.plot_checkerboard(f.generate_checkerboard_matrix(50))
new_patterns = f.store_checkerboard(patterns,board)

weights = f.hebbian_weights(patterns)
for i in range(np.size(patterns,0)):
    if ((patterns[i] != new_patterns[i]).any()):
        list_patterns = f.dynamics(patterns[i], f.hebbian_weights(patterns),20,1000)
        new_patterns[i] = list_patterns[-1][:]

for state in list_patterns:
    state = np.reshape(state, (50,50))

f.save_video(list_patterns, out_path)


new_patterns = f.store_checkerboard(patterns,board)
for i in range(np.size(patterns,0)):
    if ((patterns[i] != new_patterns[i]).any()):
        list_patterns = f.dynamics_async(patterns[i], f.hebbian_weights(patterns),30000,10000,1000)
        new_patterns[i] = list_patterns[-1][:]

for state in list_patterns:
    state = np.reshape(state, (50,50))

f.save_video(list_patterns, out_path)
 

 
