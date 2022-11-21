import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import os 



def main(patterns, nbr_perturb, function, matrix, max_iter, convergence_iter) :
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
        if function == 'dynamics' :
            P_iter = dynamics(P_perturbed, matrix,max_iter)
        elif function == 'dynamics_async' :
            P_iter = dynamics_async(P_perturbed, matrix,max_iter,convergence_iter)
        else :
            print('the parameter "function" can only be "dynamics" or "dynamics_asyn" -> pay attention to spelling ')
        pattern_converged = P_iter[-1]
        if pattern_match(patterns,pattern_converged) == None :
            answer += 1

    if answer != 0:
        print('the test of',function,'have', answer, 'differences')
    elif answer == 0:
        print('the test of',function, ' passed, there are 0 differences')


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
    
    Examples 
    -------------------------------------------------------------------
    >>> (generate_patterns(1,1))*(generate_patterns(1,1))
    array([[1]])

    """
    if (num_patterns or pattern_size) < 0:
        raise ValueError("The parameters must be positive ! ")

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
    
    Raises
    ------------------------------------------------------------------
    ValueError
        if an num_pattern is greater than the size of the pattern
    
    Examples 
    -------------------------------------------------------------------
    >>> perturb_pattern(np.array([1,-1,-1]),1)
    array([1,1,-1]) or array([-1,-1,-1]) or array([1,-1,1])
    
    """
    if num_perturb > pattern.shape[0]:
        raise ValueError("The number of perturbation must be smaller than the size of the pattern")

    i = 0
    pattern_changed = pattern.copy()
    random_position = np.zeros(pattern_changed.shape[0])
    while i < num_perturb:
        position = np.random.randint(0,np.size(pattern))
        while random_position[position] != 0 :
            position = np.random.randint(0,np.size(pattern))
        if pattern_changed[position] == 1:
            pattern_changed[position]=-1
            random_position[position] = -1
        else :
            pattern_changed[position]=1
            random_position[position] = 1
        i+=1
    return pattern_changed

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

    Raises
    ------------------------------------------------------------------
    AttributeError
        if the number of column of the pattern is not the same as the column of the memorized_pattern
    
    Examples
    ----------
    >>> pattern_match(np.array([[1,1,-1,-1],[1,1,-1,1],[-1,1,-1,1]]),np.array([1,1,-1,1]))
    1

    """
    for i in range(np.size(memorized_patterns, 0)):
        if (pattern == memorized_patterns[i]).all():
            return i 
 
def hebbian_weights(patterns):
    """
    The function takes in a matrix of patterns and returns a matrix of weights
    
    Parameters
    -----------
    patterns: a matrix of patterns, where each row is a pattern
    
    Return
    ----------
    numpy array 
    The weights matrix with the habbian weights

    Examples
    ----------
    >>> hebbian_weights(np.array([[1,1,-1,-1],[1,1,-1,1],[-1,1,-1,1]]))
    array([[ 0.        ,  0.33333333, -0.33333333, -0.33333333],
       [ 0.33333333,  0.        , -1.        ,  0.33333333],
       [-0.33333333, -1.        ,  0.        , -0.33333333],
       [-0.33333333,  0.33333333, -0.33333333,  0.        ]])
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
    new_state = np.dot(weights,state.copy())
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
    w_row = weights[position].copy()
    new_value = np.inner(w_row,new_state) #on a mis np.inner hier avec toi mais a la base c'etait np.dot
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
    states_list = np.zeros((max_iter+1,state.shape[0]))
    states_list[0] = previous_state
    for i in range(max_iter):
        new_state = update(previous_state,weights)
        states_list[i+1]=new_state.copy()
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
    states_list=[state.copy()]
    previous_state = state.copy()
    for i in range(max_iter):
        new_state = update_async(previous_state,weights)
        states_list.append(new_state)
        if(new_state == previous_state).all() :
            conv_iter += 1
            if(conv_iter == convergence_num_iter) :
                break
        else :
            conv_iter = 0
        previous_state = new_state 
    return states_list


def storkey_weights(patterns):
    """
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
    >>> storkey_weights(np.array([[1,1,-1,-1], [1,1,-1,1], [-1,1,-1,1]]))
    array([[1.125,0.25,-0.25,-0.5],[0.25,0.625,-1,0.25],[-0.25,-1,0.625,-0.25],[-0.5,0.25,-0.25,1.125]])
    """
    N = np.size(patterns[0])
    M = np.shape(patterns)[0]
    W = np.zeros((N,N))
    W_previous = np.zeros((N,N))
    H = np.zeros((N,N))
    for mu in range(M):
        patterns_matrix = np.tile(patterns[mu],(N,1)).T
        patterns_matrix_wo_diagonal = patterns_matrix.copy()
        np.fill_diagonal(patterns_matrix_wo_diagonal, 0)
        W_prev_diag = W_previous.copy()
        np.fill_diagonal(W_prev_diag, 0)
        patterns_matrix_diag = np.diag(np.diag(patterns_matrix))
        H = np.matmul(W_prev_diag, patterns_matrix_wo_diagonal)
        
        H_x_diag_patterns = np.matmul(H.copy(), patterns_matrix_diag)
        patterns_matrix_x_p_diag = np.matmul(patterns_matrix, patterns_matrix_diag)
        W = W_previous + (1/N)*(patterns_matrix_x_p_diag - (H_x_diag_patterns + H_x_diag_patterns.T))
        W_previous = W.copy()
    
    return W

def energy(state, weights):
    """
    this function calculate the energy of a given state with a formula that uses the weights matrix
    
    Parameters
    ----------
    state : array
            the state of a pattern in the network
    weights : array
            the weights matrix calculate either with the hebbian rule or the storkey rule
    
    Return
    ------
    float
        the value of the energy
    """
    E= -0.5 * np.dot(state, np.dot(weights,state))
    return E
 
def hebbian_plot_energy_time(network_state_list, weights):
    """ 
    it plots the graph of energy for a list of network state
    Parameters
    ----------
    network_state_list : array
        all the state of the network
    weights: array
        the weights matrix calculate either with the hebbian rule 
    Return
    ------
    none
    """
    E_values = np.zeros(len(network_state_list))
    x = np.arange(0,len(network_state_list))

   
    for i in range(len(network_state_list)):
        E = energy(network_state_list[i], weights)
        E_values[i] = E

    
    plt.plot(x, E_values,'g')
    plt.title("time-energy plot for hebbian weights: ")
    plt.ylabel('energy')
    plt.xlabel('time')
    plt.savefig("HebbianEnergy.png")
    plt.show()
 

def storkey_plot_energy_time(network_state_list, weights):

    """ 
    it plots the graph of energy for a list of network state
    Parameters
    ----------
    network_state_list : array
        all the state of the network
    weights: array
        the weights matrix calculate either with the storkey rule 
    Return
    ------
    none
    """
    E_values = np.zeros(len(network_state_list))
    x = np.arange(0,len(network_state_list))

   
    for i in range(len(network_state_list)):
        E = energy(network_state_list[i], weights)
        E_values[i] = E


    plt.plot(x,E_values,'b')
    plt.ylabel('energy')
    plt.xlabel('time')
    plt.title("time-energy plot for storkey weights: ")
    plt.savefig("StorkeyEnergy.png")
    plt.show()




def generate_checkerboard(checkerboard_dimension): 

    """
    it generate the matrix of 1 and -1 in which 5x5 sub-matrices of -1 or 1 are alternated
    and it plots it and save the figure

    Parameters
    ----------
    checkerboard_dimension : int
        dimension of the matrix obtained

    Return
    ------
    array
        the matrix which look like a checkerboard

    Notes
    -----
    checkerboard_dimension MUST be a multiple of 5

    Examples
    --------
    >>> generate_checkerboard(10)
    array([[ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.],
           [ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.],
           [ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.],
           [ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.],
           [ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.],
           [-1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1.],
           [-1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1.],
           [-1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1.],
           [-1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1.],
           [-1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1.]])

    >>> generate_checkerboard(15)
    array([[ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,
             1.,  1.],
           [ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,
             1.,  1.],
           [ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,
             1.,  1.],
           [ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,
             1.,  1.],
           [ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,
             1.,  1.],
           [-1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.,
            -1., -1.],
           [-1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.,
            -1., -1.],
           [-1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.,
            -1., -1.],
           [-1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.,
            -1., -1.],
           [-1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.,
            -1., -1.],
           [ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,
             1.,  1.],
           [ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,
             1.,  1.],
           [ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,
             1.,  1.],
           [ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,
             1.,  1.],
           [ 1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,
             1.,  1.]])
   

    
    """
    if checkerboard_dimension <= 0 or checkerboard_dimension%5 !=0 or type(checkerboard_dimension)!=int :
        raise TypeError("Only positive integers are allowed, which must be a multiple of 5")


    white = np.ones((5,5))
    black = np.ones((5,5)) * -1
 
        #create a lign white, black...
    lign_wb = white
    i=2
    while i<=int(checkerboard_dimension/5):
        if i%2 ==0:
            lign_wb= np.append(lign_wb, black,axis=1)
            i+=1
        else :
            lign_wb= np.append(lign_wb, white, axis=1)
            i+=1
    np.reshape(lign_wb, (5*checkerboard_dimension))        
 
        #create a lign black, white...
    lign_bw = black
    i=2
    while i<=int(checkerboard_dimension/5):
        if i%2 ==0:
            lign_bw= np.append(lign_bw, white,axis=1)
            i+=1
        else :
            lign_bw= np.append(lign_bw, black,axis=1)
            i+=1
    np.reshape(lign_bw, (5*checkerboard_dimension))  
 
        #create a checkerboard starting with a lign white, black
    checkerboard = lign_wb
    i=2
    while i<=int(checkerboard_dimension/5):
        if i%2 ==0:
            checkerboard= np.append(checkerboard, lign_bw, axis=0)
            i+=1
        else :
            checkerboard= np.append(checkerboard, lign_wb, axis=0)
            i+=1
    np.reshape(checkerboard, (checkerboard_dimension*checkerboard_dimension))  
      
    plt.figure()  
    plt.imshow(checkerboard, cmap='gray')
    plt.savefig("checkerboard.png")
    plt.show()

    return checkerboard

def flatten_checkerboard(checkerboard): 
    """ 
    it flattens the checkerboard matrix of 1 and -1 to put it in only one line.

    Parameters
    ----------
    checkerboard : array
        a matrix of 1 and -1 

    Return
    ------
    array
        the pattern obtained of 1 and -1 corresponding to this flatten checkerboard


    Examples
    --------

    >>> flatten_checkerboard(np.array([[-1,  1, -1,  1],[1, -1,  1, -1], [-1,  1, -1,  1]]))
    array([-1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1])

    >>> flatten_checkerboard(np.array([[-1,1,-1,1],[1,-1,1,-1],[-1,1,-1,1], [1,-1,1,-1]]))
    array([-1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1])
    """
    pattern = checkerboard.flatten()

    return pattern
 
def store_checkerboard(patterns, checkerboard):
    """ 
    it replaces randomly the pattern associated to the checkerboard in the matrix of patterns, after flattening it 

    Parameters
    ----------
    patterns : array
        the matrix of all patterns
    checkerboard : array
        a matrix of 1 and -1

    Return
    ------
    array
        the matrix of all patterns where we introduce the checkerboard pattern in a random position

    Notes
    -----
    the number of columns in patterns must be equal to the number of ligns in checkerboard multiplied by the number of columns in checkerboard

    Examples
    --------

    >>> store_checkerboard(np.array([[1,1,-1,-1]]), np.array([-1,1,-1,1]))
    array([[-1,  1, -1,  1]])

    """
    replacing_pattern = flatten_checkerboard(checkerboard)
    position = np.random.randint(0,np.size(patterns,0)) 
    patterns[position]= replacing_pattern
   
    return patterns
 
def reshape_states(state_list):
    """
    it reshapes every state of the list into an array of shape 50x50

    Parameters
    ----------
    state_list: array
        the list of each state of the system until convergence, state are arrays of 1 or -1

    Return
    ------
    array 
        the list with reshaped states of size 50x50

    Examples
    -------
    >>> np.shape(reshape_states(generate_patterns(2,2500)))
    (2, 50, 50)

    Note
    ----
    this function doesn't return the shape of new_shape_list, it retruns new_shape_list. But the smallest size of this list is (1,2500) 
    it is not possible to add an array of this size in the examples.
    This is why i find relevant to show the size of new_shape_list as an example

    """
    new_shape_list=([])
    for i in range(len(state_list)):
        state_list_sync_new_shape = state_list[i].reshape((50,50))
        new_shape_list.append(state_list_sync_new_shape) 

    return new_shape_list

def save_video(state_list, out_path): 
    """ 
    it creates and save a video as a gif, by creating a board for each state of the list state_list

    Parameters
    ----------
    state_list : array
        the list of all states
    out_path : array
        the path were we want to save our video

    Return
    ------
    None

    Notes
    -----
    The video should converge to the pattern of a checkerboard, this depends on the list of states
    """
    frames=[]
    fig=plt.figure()
   
    for i in range(len(state_list)):
        frames.append([plt.imshow(state_list[i], cmap='gray',animated=True)])
        
    anim = animation.ArtistAnimation(fig, frames, blit=True, repeat_delay=5000)
    writer = PillowWriter(fps=5)
    anim.save(out_path, writer=writer) 
    plt.show()



































































""