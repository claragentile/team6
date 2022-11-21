import numpy as np
import functions as f
import doctest
#import main 
#import pytest
import random
import sys
import os


random_row = np.random.randint(2, 10)
random_column = np.random.randint(2, 10)


def test_generate_patterns():
    
    patterns = f.generate_patterns(random_row,random_column)

    #verification shape of the matrix of patterns
    assert np.shape(patterns) == (random_row,random_column)

    #verification that the element of the matrix is only 1 or -1
    for i in range(np.shape(patterns)[0]):
        for j in range(np.shape(patterns)[1]):
            assert patterns[i][j] == -1 or 1

def test_perturb_pattern() :
    patterns = f.generate_patterns(1,random_column)
    pattern = patterns[0]
    num_perturb = np.random.randint(1,random_column)

    #number of perturbation check
    pattern_perturbed = f.perturb_pattern(pattern,num_perturb)
    x = 0
    for i in range(np.shape(pattern)[0]):
        if pattern[i] != pattern_perturbed[i] :
            x += 1
    assert x == num_perturb  

    #check that all the perturbations are always 1 or -1
    for i in range(np.shape(pattern_perturbed)[0]):
            assert pattern_perturbed[i] == -1 or 1
    
def test_pattern_match():

    #check if the return is correct
    patterns = f.generate_patterns(random_row,random_column)
    random = np.random.randint(1,random_row)
    random_pattern = patterns[random]
    ligne = f.pattern_match(patterns,random_pattern)
    assert ligne == random


def test_hebbian_weights() :
    patterns = f.generate_patterns(random_row,random_column)
    weights = f.hebbian_weights(patterns)

    #verfication 0 on the diag
    assert (np.diag(weights)).all() == 0

    #size check
    assert np.shape(weights) == (random_column,random_column)

    #symmetry check
    assert (weights == np.transpose(weights)).all() 

    #range check values in [-1,1]
    assert (weights >= -1).all() and (weights <= 1).all()

def test_update() :
    #check that sigma function works 
    patterns = f.generate_patterns(random_row,random_column)
    random_position = np.random.randint(1,random_row)
    weights = f.hebbian_weights(patterns)
    pattern = patterns[random_position]
    pattern_updated = f.update(pattern, weights)
    test = True
    for i in range(np.shape(pattern_updated)[0]):
        if pattern_updated[i] != -1 or 1 :
            test == False
    assert test


def test_update_async():
    #check if the return is -1 or 1
    patterns = f.generate_patterns(random_row,random_column)
    random_position = np.random.randint(1,random_row)
    weights = f.hebbian_weights(patterns)
    pattern = patterns[random_position]
    pattern_updated = f.update_async(pattern, weights)
    test = True
    for i in range(np.shape(pattern_updated)[0]):
        if pattern_updated[i] != -1 or 1 :
            test == False
    assert test

def test_dynamics() :
    patterns = f.generate_patterns(80,1000)
    weights = f.hebbian_weights(patterns)
    random_position = np.random.randint(1,80)
    pattern = patterns[random_position]
    P_perturbed = f.perturb_pattern(pattern,200)
    P_iter = f.dynamics(P_perturbed, weights,20)
    pattern_converged = P_iter[-1]
    assert f.pattern_match(patterns,pattern_converged) == None 

def test_dynamics() :
    patterns = f.generate_patterns(80,1000)
    weights = f.hebbian_weights(patterns)
    random_position = np.random.randint(1,80)
    pattern = patterns[random_position]
    P_perturbed = f.perturb_pattern(pattern,200)
    P_iter = f.dynamics_async(P_perturbed, weights,30000,10000)
    pattern_converged = P_iter[-1]
    assert f.pattern_match(patterns,pattern_converged) == None 


def test_functions_doctest():
    #testing that the examples in the docstrings are correct------------------------------------------------------------------------------------------------------------------------------------------------------------
    assert doctest.testmod(f, raise_on_error=True)


def test_generate_checkerboard():
    dimension = 5 
    checkerboard= f.generate_checkerboard(dimension)

    #testing that the dimension of the checkerboard is a multiple of 5-------------------------------------------------------------------------------------------------------------------------------------------------
    assert ((np.size(checkerboard,0)) %5 == 0) and ((np.size(checkerboard,1)) %5 == 0)

    #testing that the dimension is an integer and that the matrix is squared------------------------------------------------------------------------------------------------------------------------------------------------
    assert (type(np.size(checkerboard,0)) == int ) and (type(np.size(checkerboard,1)) == int ) and (np.size(checkerboard,0)) == (np.size(checkerboard,1))

    #testing that every value of the checkerboard matrix is 1 or -1--------------------------------------------------------------------------------------------------------------------------------------------------------
    for i in range(len(checkerboard)):
        for j in range(np.size(checkerboard,1)):
            assert (checkerboard[i][j] == -1 or 1 )

    assert np.allclose(f.generate_checkerboard(5), np.array([[1., 1., 1., 1., 1.],[1., 1., 1., 1., 1.],[1., 1., 1., 1., 1.],[1., 1., 1., 1., 1.],[1., 1., 1., 1., 1.]]))


def test_flatten_checkerboard():
    dimension = np.random.randint(1,4)
    dimension = dimension *5 
    pattern = f.flatten_checkerboard(f.generate_checkerboard(dimension))

    #testing if the checkerboard patten has the dimension of a flattened pattern ----------------------------------------------------------------------------------------------------------------------------------------------------
    assert ((np.size(pattern,0) )== dimension*dimension)

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
    assert np.size(patterns, 1) == len(f.flatten_checkerboard(checkerboard))

    #testing if the checkerboard pattern has been introduce to the patterns matrix----------------------------------------------------------------------------------------------------------------------------------------
    assert (f.pattern_match(new_patterns, f.flatten_checkerboard(checkerboard)) != None)


def test_convergence():
    
    patterns = f.generate_patterns(50,2500)
    checkerboard = f.generate_checkerboard(50)
    new_patterns = f.store_checkerboard(patterns,checkerboard)


    flattened_checkerboard= f.flatten_checkerboard(checkerboard)
    checkerboard_index = f.pattern_match(new_patterns, flattened_checkerboard)

    checkerboard_perturbed = f.perturb_pattern(new_patterns[checkerboard_index], 1000)

    #testing is the system has converge to the checkerboard pattern starting with a pattern different from a checkerboard--------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #synchronous rule and hebbian weights
    state_list_1 = f.dynamics(checkerboard_perturbed, f.hebbian_weights(patterns),20)
    assert (state_list_1[0]!= flattened_checkerboard).any() and (state_list_1[-1]== flattened_checkerboard).all()

    #asynchronous rule and hebbian weights
    state_list_2 = f.dynamics_async(checkerboard_perturbed, f.hebbian_weights(patterns),30000,10000)
    assert (state_list_2[0]!= flattened_checkerboard).any() and (state_list_2[-1]== flattened_checkerboard).all()

    #synchronous rule and storkey weights
    state_list_3 = f.dynamics(checkerboard_perturbed, f.storkey_weights(patterns),20)
    assert (state_list_2[0]!= flattened_checkerboard).any() and (state_list_3[-1]== flattened_checkerboard).all()

    #asynchronous rule and storkey weights
    state_list_4 = f.dynamics_async(checkerboard_perturbed, f.storkey_weights(patterns), 30000, 10000) 
    assert (state_list_2[0]!= flattened_checkerboard).any() and (state_list_4[-1]== flattened_checkerboard).all()


def test_save_video():

    patterns = f.generate_patterns(50,2500)
    state=f.generate_patterns(1,2500)
    state_list = f.dynamics(state, f.hebbian_weights(patterns),20)

    new_shape_list = f.reshape_states(state_list)

    #testing if the list of states has the correct size : we should have a certain number of states of size 50x50 --------------------------------------------------------------------------------------------------------------------------
    assert new_shape_list.shape[1] == new_shape_list.shape[2] and new_shape_list.shape[1] == 50
    
    #testing if the path exists on my computer-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    out_path = os.path.join("C:\\Users\\33645\\Desktop\\prog\\projet\\BIO-210-22-team-6" , "video.gif")
    assert os.path.exists(out_path)
