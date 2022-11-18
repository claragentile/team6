
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import os 
"""
print('hello')
# test dynamics-----------------------------------------------------------------
#patterns = f.generate_patterns(80,1000)

#profiled = "f.test_main(patterns,200,'dynamics','hebbian',20,0)"
#cProfile.run(profiled,sort="cumtime")


#f.test_main(patterns,200,'dynamics','hebbian',20,0)

# test dynamics_async-----------------------------------------------------------------

#patterns = f.generate_patterns(80,1000) 

#f.test_main(patterns,200,'dynamics_async','hebbian',20000,3000)

#test of storkey dynamics -------------------------------------------------------------------------------------------------------

patterns = f.generate_patterns(80,1000)

f.test_main(patterns,200,'dynamics','storkey',20,0)

#test of storkey dynamics_async -------------------------------------------------------------------------------------------------------

patterns = f.generate_patterns(80,1000) 

f.test_main(patterns,200,'dynamics_async','storkey',20000,3000)
"""
"""
#test of energy--------------------------------------------------------------------------------------------------
 
patterns = f.generate_patterns(50, 2500)
position = np.random.randint(0,np.size(patterns,0)) 
perturbed_pattern = f.perturb_pattern(patterns[position], 1000) 

network_state_list_h_s = f.dynamics(perturbed_pattern, f.hebbian_weights(patterns), 20)
#network_state_list_s_s = f.dynamics(perturbed_pattern, f.storkey_weights(patterns), 20)
 
network_state_list_h_a = f.dynamics_async(perturbed_pattern, f.hebbian_weights(patterns), 30000,10000)
#network_state_list_s_a = f.dynamics_async(perturbed_pattern, f.storkey_weights(patterns), 30000,10000)
 
f.hebbian_plot_energy_time(network_state_list_h_s, f.hebbian_weights(patterns))
#f.hebbian_plot_energy_time(network_state_list_h_a, f.hebbian_weights(patterns))
#f.storkey_plot_energy_time(network_state_list_s_s, f.strokey_weights(patterns))
#f.storkey_plot_energy_time(network_state_list_s_a, f.strokey_weights(patterns))

"""
#-test of visualization-------------------------------------------------------------------------------------------------------------
patterns = f.generate_patterns(50,2500)
board= f.plot_checkerboard(f.generate_checkerboard_matrix(50))
new_patterns = f.store_checkerboard(patterns,board)

weights = f.hebbian_weights(new_patterns)


checkerboard_index = f.pattern_match(new_patterns, f.flatten_checkerboard(board))

checkerboard_perturbed = f.perturb_pattern(new_patterns[checkerboard_index], 1000)

state_list_sync = f.dynamics(checkerboard_perturbed, f.hebbian_weights(patterns),20)
#state_list_async = f.dynamics_async(checkerboard_perturbed, f.hebbian_weights(patterns), 30000, 10000) #store every 1000??

new_shape_list=([])
for i in range(len(state_list_sync)):
    state_list_sync_new_shape = state_list_sync[i].reshape((50,50))
    new_shape_list.append(state_list_sync_new_shape) 

out_path_sync = os.path.join("C:\\Users\\33645\\Desktop\\prog\\projet\\BIO-210-22-team-6" , "video_sync.gif")
f.save_video(new_shape_list, out_path_sync)
