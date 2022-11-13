import numpy as np
 
#generate a 2d array in which each row is a pattern
def generate_patterns(num_patterns, pattern_size): #M,N
    patterns = np.random.choice([-1,1], (num_patterns,pattern_size)) 
    return patterns 

#perturb some element in a given pattern
def perturb_pattern(pattern, num_perturb):
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
    for i in range(np.size(memorized_patterns, 0)):
        if (pattern == memorized_patterns[i]).all():
            return i 
 
def hebbian_weights(patterns):
    weights_matrix= (1/np.shape(patterns)[0])*(np.matmul(np.transpose(patterns),patterns))-np.identity(np.shape(patterns)[1])
    return weights_matrix
 

def update(state, weights):
    #state is a pattern
    new_state = np.dot(weights,state)
    for i in range(np.size(new_state)):
        if new_state[i] >= 0:
            new_state[i] = 1
        else :
            new_state[i] = -1
    return new_state
 
def update_async(state, weights):
    new_value = np.dot(weights,state)
    if new_value >= 0:
            new_value = 1
    else :
        new_value = -1
    return new_value
'''
def update_async(state, weights):
    new_state = state.copy()
    position = np.random.randint(0,np.size(new_state))
    w_row = weights[position]
    #new_state[position]= np.dot(w_row, state)
    print('new_state[position] : ',new_state[position])
    new_state[position] = update(new_state,w_row)
    print(' nouvelle new_state[position] : ',new_state[position])
    return new_state
'''
def dynamics(state, weights, max_iter):
    #new_state = state.copy() #il faut enlever ca sinon ca marche pas
    perturbed_state = perturb_pattern(state.copy(), 200)
    new_state = update(perturbed_state,weights)
    states_list = new_state
    for i in range(max_iter):
        states_list = np.vstack([states_list,new_state])
        if (new_state == state).all() :
            break
        
        perturbed_state = new_state
        new_state = update(perturbed_state, weights)
    return states_list

def dynamics_async(state, weights, max_iter, convergence_num_iter):
    perturbed_state = perturb_pattern(state.copy(), 200)
    iter = 0
    states_list = perturbed_state.copy()
    for i in range(max_iter):
        position = np.random.randint(0,np.size(perturbed_state))
        w_row = weights[position]
        if(state[position]==perturbed_state[position]) :
            iter += 1
            if(iter == convergence_num_iter) :
                break
        else :
            iter = 0
            perturbed_state[position] = update_async(perturbed_state,w_row)
            states_list = np.vstack([states_list,perturbed_state])
    return states_list



#def storkey_weights(patterns):
def storkey_weights(patterns):
    N = np.size(patterns[0])
    M = np.shape(patterns)[0]
    W = np.zeros((N,N))
    W_previous = np.zeros((N,N))
    # maybe don't need the mu and you can do with when you do the steps
    H = np.zeros((N,N))
    for mu in range(M):
        H = np.matmul(W_previous - np.diag(np.diag(W_previous)), np.tile(patterns[mu].T,(N,1)) - np.diag(np.diag(np.tile(patterns[mu].T,(N,1)))))
        W = W_previous + (1/N)*(np.matmul(np.tile(patterns[mu].T,(N,1)), np.diag(np.diag(np.tile(patterns[mu].T,(N,1))))) - (np.matmul(H, np.diag(np.diag(np.tile(patterns[mu].T,(N,1))))) + (np.matmul(H, np.diag(np.diag(np.tile(patterns[mu].T,(N,1)))))).T))
        W_previous = W
    return W