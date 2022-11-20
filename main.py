
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import os 


"""
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


#test of energy----------------------------------------------------------------------------------------------------------------------------------
 
patterns = f.generate_patterns(50, 2500)
position = np.random.randint(0,np.size(patterns,0)) 
perturbed_pattern = f.perturb_pattern(patterns[position], 1000) 

#hebbian synchronous---------------------------------------------------------------------------------------------------------------------------
#network_state_list_h_s = f.dynamics(perturbed_pattern, f.hebbian_weights(patterns), 20)
#f.hebbian_plot_energy_time(network_state_list_h_s, f.hebbian_weights(patterns))

#storkey synchronous---------------------------------------------------------------------------------------------------------------------------------------
#network_state_list_s_s = f.dynamics(perturbed_pattern, f.storkey_weights(patterns), 20)
#f.storkey_plot_energy_time(network_state_list_s_s, f.storkey_weights(patterns))
 
#hebbian asynchronous------------------------------------------------------------------------------------------------------------------------------------------------
#network_state_list_h_a = f.dynamics_async(perturbed_pattern, f.hebbian_weights(patterns), 30000,10000)
#f.hebbian_plot_energy_time(network_state_list_h_a, f.hebbian_weights(patterns))

#storkey asynchronous-------------------------------------------------------------------------------------------------------------------------------------------------
#network_state_list_s_a = f.dynamics_async(perturbed_pattern, f.storkey_weights(patterns), 30000,10000)
#f.storkey_plot_energy_time(network_state_list_s_a, f.storkey_weights(patterns))



#-test of visualization-------------------------------------------------------------------------------------------------------------
patterns = f.generate_patterns(50,2500)
checkerboard = f.generate_checkerboard(50)
new_patterns = f.store_checkerboard(patterns,checkerboard)
weights = f.hebbian_weights(new_patterns)
checkerboard_index = f.pattern_match(new_patterns, f.flatten_checkerboard(checkerboard))
checkerboard_perturbed = f.perturb_pattern(new_patterns[checkerboard_index], 1000)

#test of sync hebbian ------------------------------------------------------------------------------------------------------
state_list = f.dynamics(checkerboard_perturbed, f.hebbian_weights(patterns),20)

#test of async hebbian -------------------------------------------------------------------------------------------------------
#state_list = f.dynamics_async(checkerboard_perturbed, f.hebbian_weights(patterns),30000,10000)

#test of sync storkey----------------------------------------------------------------------------------------------------------
#state_list = f.dynamics(checkerboard_perturbed, f.storkey_weights(patterns),20)

#test of async storkey ------------------------------------------------------------------------------------------------
#state_list = f.dynamics_async(checkerboard_perturbed, f.storkey_weights(patterns), 30000, 10000) 

out_path = os.path.join("C:\\Users\\33645\\Desktop\\prog\\projet\\BIO-210-22-team-6" , "video.gif")

f.save_video(f.reshape_states(state_list), out_path)
