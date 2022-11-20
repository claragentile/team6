import numpy as np
import functions as f
import doctest
import random
import os

def test_functions_doctest():
    #testing that the examples in the docstrings are correct------------------------------------------------------------------------------------------------------------------------------------------------------------
    assert doctest.testmod(f, raise_on_error=True)

def test_generate_checkerboard():
    dimension = np.random.randint(1,4)
    dimension = dimension *5 
    checkerboard= f.generate_checkerboard(dimension)

    #testing that the dimension of the checkerboard is a multiple of 5-------------------------------------------------------------------------------------------------------------------------------------------------
    assert ((np.size(checkerboard,0)) %5 == 0) and ((np.size(checkerboard,1)) %5 == 0)

    #testing that the dimension is an integer and that the matrix is squared------------------------------------------------------------------------------------------------------------------------------------------------
    assert (type(np.size(checkerboard,0)) == int ) and (type(np.size(checkerboard,1)) == int ) and (np.size(checkerboard,0)) == (np.size(checkerboard,1))

    #testing that every value of the checkerboard matrix is 1 or -1--------------------------------------------------------------------------------------------------------------------------------------------------------
    for i in range(len(checkerboard)):
        for j in range(np.size(checkerboard,1)):
            assert (checkerboard[i][j] == -1 or 1 )


def test_flatten_checkerboard():
    dimension = np.random.randint(1,4)
    dimension = dimension *5 
    pattern = f.flatten_checkerboard(f.generate_checkerboard(dimension))

    #testing if the checkerboard patten has the dimension of a flattened pattern ----------------------------------------------------------------------------------------------------------------------------------------------------
    assert ((np.size(pattern,0) )== 1) and ((np.size(pattern,1)) == dimension*dimension)

    #testing if every element of the pattern is 1 or -1 --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    for i in range(len(pattern)):
        assert(pattern[i]== -1 or 1)


def test_store_checkerboard():
    dimension = np.random.randint(1,4)
    dimension = dimension *5 
    checkerboard = f.generate_checkerboard(dimension)
    patterns = f.generate_patterns(np.random.randint(1,4),dimension*dimension)

    new_patterns = f.store_checkerboard(patterns, checkerboard)

    #testing if the shape of the checkerboard is compatible to the shape of patterns-------------------------------------------------------------------------------------------------------------------------------------------------
    assert np.size(patterns, 1)== np.size(f.flatten_checkerboard(checkerboard), 1)

    #testing if the checkerboard pattern has been introduce to the patterns matrix----------------------------------------------------------------------------------------------------------------------------------------
    assert (f.pattern_match(patterns, f.flatten_checkerboard(checkerboard)) != 0)


def test_conergence():
    
    patterns = f.generate_patterns(50,2500)
    checkerboard = f.generate_checkerboard(50)
    new_patterns = f.store_checkerboard(patterns,checkerboard)

    weights = f.hebbian_weights(new_patterns)

    flattened_checkerboard= f.flatten_checkerboard(checkerboard)
    checkerboard_index = f.pattern_match(new_patterns, flattened_checkerboard)

    checkerboard_perturbed = f.perturb_pattern(new_patterns[checkerboard_index], 1000)

    #testing is the system has converge to the checkerboard pattern starting with a pattern different from a checkerboard--------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #synchronous rule and hebbian weights
    state_list_1 = f.dynamics(checkerboard_perturbed, f.hebbian_weights(patterns),20)
    assert (state_list_1[0]!= flattened_checkerboard) and (state_list_1[-1]== flattened_checkerboard)

    #asynchronous rule and hebbian weights
    state_list_2 = f.dynamics_async(checkerboard_perturbed, f.hebbian_weights(patterns),30000,10000)
    assert (state_list_2[0]!= flattened_checkerboard) and (state_list_2[-1]== flattened_checkerboard)

    #synchronous rule and storkey weights
    state_list_3 = f.dynamics(checkerboard_perturbed, f.storkey_weights(patterns),20)
    assert (state_list_2[0]!= flattened_checkerboard) and (state_list_3[-1]== flattened_checkerboard)

    #asynchronous rule and storkey weights
    state_list_4 = f.dynamics_async(checkerboard_perturbed, f.storkey_weights(patterns), 30000, 10000) 
    assert (state_list_2[0]!= flattened_checkerboard) and (state_list_4[-1]== flattened_checkerboard)



def test_save_video():

    patterns = f.generate_patterns(50,2500)
    state=f.generate_patterns(1,2500)
    state_list = f.dynamics(state, f.hebbian_weights(patterns),20)

    new_shape_list = f.reshape_states(state_list)

    #testing if the list of states has the correct size : we should have a certain number of states of size 50x50 --------------------------------------------------------------------------------------------------------------------------
    assert new_shape_list.shape[1] == new_shape_list.shape[2] and new_shape_list.shape[1] == 50
    
    #testing if the path exists on my computer-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    out_path = os.path.join("C:\\Users\\33645\\Desktop\\prog\\projet\\BIO-210-22-team-6" , "video.gif")
    os.path.exists(out_path)
    

    
    



    






    


