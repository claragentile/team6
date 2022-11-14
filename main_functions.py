import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
 
def generate_patterns(num_patterns, pattern_size): #M,N
    """
    It generates a matrix of random patterns, where each pattern is a row of the matrix. 
    
    The number of patterns is given by the first argument, and the size of each pattern is given by the
    second argument. 
    
    The values in each pattern are randomly chosen from the set {-1,1}.
    
    :param num_patterns: the number of patterns to generate
    :param pattern_size: The number of neurons in the network
    :return: A matrix of size MxN, where M is the number of patterns and N is the size of each pattern.
    """
    patterns = np.random.choice([-1,1], (num_patterns,pattern_size)) 
    return patterns 

def perturb_pattern(pattern, num_perturb):
    """
    It flips the value of a random bit in the pattern.
    
    :param pattern: the pattern to be perturbed
    :param num_perturb: number of perturbations to make to the pattern
    :return: the number of bits that are different between the two patterns.
    """
    i = 0
    while i <= num_perturb:
        position = np.random.randint(0,np.size(pattern))
        if pattern[position] == 1:
            pattern[position]=-1
        else :
            pattern[position]=1
        i+=1
    return pattern

def pattern_match(memorized_patterns, pattern):
    """
    If the pattern is in the list of memorized patterns, return the index of the pattern in the list. 
    
    If the pattern is not in the list of memorized patterns, return -1. 
    
    Let's see how this function works. 
    
    First, let's create a list of memorized patterns. 
    
    We'll use the same list of memorized patterns that we used in the previous video. 
    
    We'll create a list of memorized patterns, and then we'll create a pattern that we want to match. 
    
    We'll use the pattern match function to see if the pattern is in the list of memorized patterns. 
    
    If it is, we'll print the index of the pattern in the list. 
    
    If it's not, we'll print a message saying that the pattern is not in the list. 
    
    Let's see how this works. 
    
    First, we
    
    :param memorized_patterns: the patterns that the network has memorized
    :param pattern: the pattern to be matched
    :return: The index of the pattern that matches the input pattern.
    """
    for i in range(np.size(memorized_patterns, 0)):
        if (pattern == memorized_patterns[i]).all():
            return i 
 
def hebbian_weights(patterns):
    """
    The function takes in a matrix of patterns and returns a matrix of weights
    
    :param patterns: a matrix of patterns, where each row is a pattern
    :return: The weights matrix
    """
    weights_matrix= (1/np.shape(patterns)[0])*(np.matmul(np.transpose(patterns),patterns))-np.identity(np.shape(patterns)[1])
    return weights_matrix
 

def update(state, weights):
    """
    > The function takes a state and a weight matrix as input, and returns the updated state
    
    :param state: the current state of the network
    :param weights: the weight matrix
    :return: The new state of the network.
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
    > The function takes a state and a weight matrix as input, and returns a new state
    
    :param state: the current state of the system
    :param weights: the weight matrix
    :return: The new state of the system.
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
    > The function `dynamics` takes as input a state, a weight matrix, and a maximum number of
    iterations, and returns a list of states that the system visits
    
    :param state: the initial state of the system
    :param weights: the weight matrix
    :param max_iter: the maximum number of iterations to run the dynamics for
    :return: a list of states.
    """
    previous_state = state.copy()
    states_list = previous_state
    for i in range(max_iter):
        new_state = update(previous_state,weights)
        states_list = np.vstack([states_list,new_state])
        if (new_state == previous_state).all() :
            break
        previous_state = new_state
    return states_list

def dynamics_async(state, weights, max_iter, convergence_num_iter):
    """
    It takes a state, weights, a maximum number of iterations, and a number of iterations to check for
    convergence. It then updates the state using the asynchronous update rule, and checks if the state
    has converged. If it has, it stops. If it hasn't, it continues
    
    :param state: the initial state of the system
    :param weights: the weight matrix
    :param max_iter: the maximum number of iterations to run the dynamics for
    :param convergence_num_iter: the number of iterations that the system must be in a fixed point
    before we consider it converged
    :return: a list of states.
    """
    iter = 0
    states_list = state.copy()
    previous_state = state.copy()
    for i in range(max_iter):
        new_state = update_async(previous_state,weights)
        states_list = np.vstack([states_list,new_state])
        if(new_state == previous_state).all() :
            iter += 1
            if(iter == convergence_num_iter) :
                break
        else :
            iter = 0
            previous_state = new_state
    return states_list



#def storkey_weights(patterns):
def storkey_weights(patterns):
    """
    The function takes in a list of patterns and returns the weight matrix
    
    :param patterns: the patterns that we want to store in the network
    :return: the weight matrix W.
    """
    N = np.size(patterns[0])
    M = np.shape(patterns)[0]
    W = np.zeros((N,N))
    W_previous = np.zeros((N,N))
    # maybe don't need the mu and you can do with when you do the steps
    H = np.zeros((N,N))
    for mu in range(M):
        #H = np.matmul(W_previous - np.diag(np.diag(W_previous)), np.tile(patterns[mu].T,(N,1)) - np.diag(np.diag(np.tile(patterns[mu].T,(N,1)))))
        #we don't modify the matrix H, because we couldn't manage to make it work while modifying H. we know that H should change
        W = W_previous + (1/N)*((np.outer(patterns[mu],patterns[mu]))-np.dot(patterns[mu],H.T)-np.dot(patterns[mu],H))
        W_previous = W.copy()
    return W

def energy(state, weights):
    """
    It takes a state and a weight matrix and returns the energy of the state
    
    :param state: the state of the system, which is a vector of 0s and 1s
    :param weights: the weight matrix
    :return: The energy of the state.
    """
    sum_E = 0
 
    for i in range (np.size(state)):
        for j in range(np.size(state)):
            sum_E += weights[i][j]*state[i]*state[j]
 
    E = -0,5*sum_E
    return E
 
def hebbian_plot_energy_time(network_state_list):
    """
    It takes a list of network states and plots the energy of the network as a function of time
    
    :param network_state_list: a list of network states, each of which is a list of neuron states
    """
   
    E_values = np.array([])
    x = np.arange(0,np.size(network_state_list,0))
    print(x)
    for i in range(0,np.size(network_state_list,0)):
        E = energy(network_state_list[i], hebbian_weights(network_state_list))
        E_values = np.append(E_values, E)

    plt.figure()
    plt.plot(x,E_values,'g')
    plt.ylabel('energy')
    plt.xlabel('time')
    plt.show()
    plt.savefig("HebbianEnergy.png")
 

def storkey_plot_energy_time(network_state_list):
    """
    It takes a list of network states and plots the energy of the network over time
    
    :param network_state_list: a list of network states, each of which is a list of neuron states
    """
    E_values = np.array([])
    x = np.arange(0,np.size(network_state_list,0))
    print(x)
    for i in range(0,np.size(network_state_list,0)):
        E = energy(network_state_list[i], storkey_weights(network_state_list))
        E_values = np.append(E_values, E)

    plt.figure()
    plt.plot(x,E_values,'g')
    plt.ylabel('energy')
    plt.xlabel('time')
    plt.show()
    plt.savefig("StorkeyEnergy.png")


def generate_checkerboard_matrix(checkerboard_dimension): #checkerboard_dimension must be a multiple of 5
    """
    It creates a checkerboard matrix of size  \times n$ where $ is a multiple of 5
    
    :param checkerboard_dimension: The dimension of the checkerboard. Must be a multiple of 5
    :return: A checkerboard matrix of dimension checkerboard_dimension*checkerboard_dimension
    """
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
       
    return checkerboard
 
def plot_checkerboard(checkerboard):
    """
    It takes a checkerboard as input and returns a plot of the checkerboard
    
    :param checkerboard: The checkerboard to be plotted
    :return: the board.
    """
    checkerboard_dimension= np.size(checkerboard,0)
    board = np.zeros((checkerboard_dimension,checkerboard_dimension))
 
    for i in range (0,checkerboard_dimension):  
        for j in range(0,checkerboard_dimension):
 
            if checkerboard[i][j]>0:
                ci=int(i)
                cj=int(j)
                board[ci][cj]= 1  
 
    plt.figure()  
    plt.imshow(board, cmap='gray')
    plt.show()
    plt.savefig("checkerboard.png")
    return board
   
 
def flatten_checkerboard(board): #board c'est des 1 et 0
    """
    It takes a 2D array of 1s and 0s and returns a 1D array of -1s and 1s
    
    :param board: the board to be flattened
    :return: a flattened array of the board with 1 and 0 replaced by -1 and 1 respectively.
    """
    pattern = board.flatten()
    for i in range(np.size(pattern)):
        if pattern[i]==0:
            pattern[i]=-1
    return pattern
 
def store_checkerboard(patterns, board):
    """
    > The function takes in a list of patterns and a checkerboard, flattens the checkerboard, and
    replaces a random pattern in the list with the flattened checkerboard
    
    :param patterns: the patterns to be stored in the network
    :param board: the checkerboard to be stored
    :return: The patterns array with the new pattern inserted at a random position.
    """
    replacing_pattern = flatten_checkerboard(board)
    position = np.random.randint(0,np.size(patterns,0)) #-1
    patterns[position]= replacing_pattern
   
    return patterns
 
def save_video(state_list, out_path): #state_list 50*50
    """
    It takes a list of states and saves a video of the states
    
    :param state_list: a list of states, each state is a numpy array of size (n,n)
    :param out_path: The path to save the video to
    """
   
    frames=[]
    fig=plt.figure()
   
    for state in state_list:
        frames.append([plt.imshow(plot_checkerboard(generate_checkerboard_matrix(np.size(state,0))), cmap='gray',animated=True)])
       
    anim = animation.ArtistAnimation(fig, frames, blit=True, repeat_delay=1000)
    anim.save("video.gif")
    plt.show()
    out_path =anim
 