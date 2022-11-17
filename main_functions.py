import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

def test_main(patterns, nbr_perturb, function, matrix, max_iter, convergence_iter) :
    """
    It takes in a set of patterns, a number of perturbations, a function, a matrix, a maximum number of
    iterations, and a convergence iteration, and then it tests the function with the matrix to see if the patterns have converged to the original pattern 
    
    Parameters
    -----------------------------------------------------------------
    patterns: the patterns you want to test
    nbr_perturb: the number of bits to flip in the pattern
    function: 'dynamics' or 'dynamics_async'
    matrix: 'hebbian' or 'storkey'
    max_iter: the number of iterations you want to run the dynamics for
    convergence_iter: the number of iterations to check for convergence
    """
    
    answer = 0 
    for i in range(np.size(patterns,0)):
        P_perturbed = perturb_pattern(patterns[i],nbr_perturb)
        if function == 'dynamics' and matrix == 'hebbian' :
            P_iter = dynamics(P_perturbed, hebbian_weights(patterns),max_iter)
        elif function == 'dynamics_async' and matrix == 'hebbian' :
            P_iter = dynamics_async(P_perturbed, hebbian_weights(patterns),max_iter,convergence_iter)
        elif function == 'dynamics' and matrix == 'storkey' :
            P_iter = dynamics(P_perturbed, storkey_weights(patterns),max_iter)
        elif function == 'dynamics_async' and matrix == 'hebbian' :
            P_iter = dynamics_async(P_perturbed, storkey_weights(patterns),max_iter,convergence_iter)
        else :
            print('you called the function wrong')
        if (P_iter[-1][:] != patterns[i]).any():
            answer += 1
    if answer != 0:
        print('the test of',function,'with', matrix ,'weights have', answer, 'differences')
    elif answer == 0:
        print('the test of',function, 'with',matrix ,'weights passed, there are 0 differences')


def generate_patterns(num_patterns, pattern_size): #M,N
    """
    It generates a matrix of random patterns, where each pattern is a row of the matrix. 
    
    The number of patterns is given by the first argument, and the size of each pattern is given by the
    second argument. 
    
    The values in each pattern are randomly chosen from the set {-1,1}.
    
    Parameters
    -----------------------------------------------------------------
    num_patterns: the number of patterns to generate
    pattern_size: The number of neurons in the network

    Return
    ------------------------------------------------------------------
    A matrix of size MxN, where M is the number of patterns and N is the size of each pattern.

    Raises
    ------------------------------------------------------------------
    ValueError
        if an argument is negative
    
    Notes
    ------------------------------------------------------------------
    ?????????

    """
    patterns = np.random.choice([-1,1], (num_patterns,pattern_size)) 
    return patterns 

def perturb_pattern(pattern, num_perturb):
    """
    It flips the value of a random bit in the pattern.
    
    Parameters
    -----------------------------------------------------------------
    pattern: the pattern to be perturbed
    num_perturb: number of perturbations to make to the pattern

    Return
    ------------------------------------------------------------------
    the number of bits that are different between the two patterns.
    
    Notes
    ------------------------------------------------------------------
    ?????????
    """
    i = 0
    random_position = np.zeros(pattern.shape[0])
    while i <= num_perturb:
        position = np.random.randint(0,np.size(pattern))
        while random_position[position] != 0 :
            position = np.random.randint(0,np.size(pattern))
        if pattern[position] == 1:
            pattern[position]=-1
            random_position[position] = -1
        else :
            pattern[position]=1
            random_position[position] = 1
        i+=1
    return pattern

def pattern_match(memorized_patterns, pattern):
    """
    If the pattern is in the list of memorized patterns, return the index of the pattern in the list. 
    If the pattern is not in the list of memorized patterns, return none. 
    
    Parameters
    -----------------------------------------------------------------
    memorized_patterns: the patterns that the network has memorized
    pattern: the pattern to be matched

    Return
    ------------------------------------------------------------------
    The index of the pattern that matches the input pattern.
    """
    for i in range(np.size(memorized_patterns, 0)):
        if (pattern == memorized_patterns[i]).all():
            return i 
 
def hebbian_weights(patterns):
    """
    The function takes in a matrix of patterns and returns a matrix of weights
    
    Parameters
    -----------------------------------------------------------------
    patterns: a matrix of patterns, where each row is a pattern

    Return
    ------------------------------------------------------------------
    The weights matrix
    """
    weights_matrix= (1/np.shape(patterns)[0])*(np.matmul(np.transpose(patterns),patterns))-np.identity(np.shape(patterns)[1])
    return weights_matrix
 

def update(state, weights):
    """
    The function takes a state and a weight matrix as input, and returns the updated state
    
    Parameters
    -----------------------------------------------------------------
    state: the current state of the network
    weights: the weight matrix
    
    Return
    ------------------------------------------------------------------
    The new state of the network.
    """
    #state is a pattern
    new_state = np.dot(weights,state)
    for i in range(np.size(new_state)):
        if new_state[i] >= 0:
            new_state[i] = 1
        else :
            new_state[i] = -1
    return new_state
 
def update_async(state, weights):
    """
    The function takes a state and a weight matrix as input, and returns a new state
    
    Parameters
    -----------------------------------------------------------------
    state: the current state of the system
    weights: the weight matrix

    Return
    ------------------------------------------------------------------
    The new state of the system with the new value at a random position.
    """
    new_state = state.copy()
    position = np.random.randint(0,np.size(state))
    w_row = weights[position]
    new_value = np.dot(w_row,new_state)
    if new_value >= 0:
            new_value = 1
    else :
        new_value = -1
    new_state[position] = new_value
    return new_state

def dynamics(state, weights, max_iter):
    """
    The function takes as input a state, a weight matrix, and a maximum number of
    iterations, and returns a list of states that the system visits
    
    Parameters
    -----------------------------------------------------------------
    state: the initial state of the system
    weights: the weight matrix
    max_iter: the maximum number of iterations to run the dynamics for

    Return
    ------------------------------------------------------------------
    a list of the historic states.
    """
    previous_state = state.copy()
    #states_list = previous_state
    states_list = np.zeros([max_iter,state.shape[0]])
    states_list[0] = previous_state.copy()
    for i in range(max_iter):
        new_state = update(previous_state,weights)
        states_list[i+1]=new_state.copy()
        #states_list = np.vstack([states_list,new_state])
        if (new_state == previous_state).all() :
            break
        previous_state = new_state
    states_list = states_list[0:i+2]
    return states_list

def dynamics_async(state, weights, max_iter, convergence_num_iter):
    """
    It takes a state, weights, a maximum number of iterations, and a number of iterations to check for
    convergence. It then updates the state using the asynchronous update rule, and checks if the state
    has converged. If it has, it stops. If it hasn't, it continues
    
    Parameters
    -----------------------------------------------------------------
    state: the initial state of the system
    weights: the weight matrix
    max_iter: the maximum number of iterations to run the dynamics for
    convergence_num_iter: the number of iterations that the system must be in a fixed point
    before we consider it converged

    Return
    ------------------------------------------------------------------
    a list of the historic states.
    """
    conv_iter = 0
    #states_list = state.copy()
    states_list = np.zeros([max_iter,state.shape[0]])
    states_list[0] = state.copy()
    previous_state = state.copy()
    for i in range(max_iter):
        new_state = update_async(previous_state,weights)
        #states_list = np.vstack([states_list,new_state])
        states_list[i+1] = new_state.copy()
        if(new_state == previous_state).all() :
            conv_iter += 1
            if(conv_iter == convergence_num_iter) :
                break
        else :
            conv_iter = 0
            previous_state = new_state
    states_list = states_list[0:i+2]
    return states_list


def storkey_weights(patterns):
    """"
    Function that returns the weight matrix, with the storkey rule.
    Parameters
    -----------------
    patterns : array of 1 or -1 that represent the activation of neurons
                each line of the array is neuron
    
    Returns
    -----------------
    the weight matrix

    Notes
    -----------------
    the patterns matrix has to be a matrix of -1 or 1, otherwise 
    you will not get the right matrix

    Examples
    -----------------
    >>> storkey_weights(np.array([[-1,1,-1], [1,1,-1]]))
    [[0.22222222 0.44444444 0.44444444]
    [0.44444444 0.22222222 0.44444444]
    [0.44444444 0.44444444 0.22222222]]

    >>> storkey_weights(np.array([[-1,1,-1,1,1], [1,1,-1,-1,1], [-1,-1,-1,1,-1]]))
    [[0.024 0.168 0.168 0.168 0.168]
    [0.168 0.024 0.168 0.168 0.168]
    [0.168 0.168 0.024 0.168 0.168]
    [0.168 0.168 0.168 0.024 0.168]
    [0.168 0.168 0.168 0.168 0.024]]

    """
    N = np.size(patterns[0])
    M = np.shape(patterns)[0]
    W = np.zeros((N,N))
    W_previous = np.zeros((N,N))
    H = np.zeros((N,N))
    for mu in range(M):
        print('mu', mu)
        W_previous_wo_diag = W_previous.copy()
        np.fill_diagonal(W_previous_wo_diag, 0)
        p_matrix = np.tile(patterns[mu],(N,1)).T
        p_matrix_wo_diag = p_matrix.copy()
        np.fill_diagonal(p_matrix_wo_diag, 0)
        p_matrix_diag = np.diag(np.diag(patterns[mu]))
        H = np.matmul(W_previous_wo_diag, p_matrix_diag)
        pxp_diag = np.matmul(p_matrix, p_matrix_diag)
        Hxp_diag = np.matmul(H, p_matrix_diag)
        W = W_previous + (1/N)*(pxp_diag - (Hxp_diag + Hxp_diag.T))
        W_previous = W.copy()

    return W




