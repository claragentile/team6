import numpy as np
 
def generate_patterns(num_patterns, pattern_size): #M,N
    patterns = np.random.choice([-1,1], (num_patterns,pattern_size))
    return patterns
 
def perturb_pattern(pattern, num_perturb):
    i = 0
    while i <= num_perturb:
        position = np.random.randint(0,np.size(pattern)-1)
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
   
    new_state = np.zeros((1,np.size(weights, 1)))
   
    #on sait  plus pq if (state == p[line]).all() :
        #break
       
    new_state = np.dot(weights, state.T)
    for i in range(np.size(state)):
        if new_state[i] >= 0:
            new_state[i] = 1
        if new_state[i] < 0:
            new_state[i] = -1
        state = new_state
    return new_state
 

def update_async(state, weights):
    new_state= state.copy()
    position = np.random.randint(0,np.size(state)-1)
    w_row= weights[position]
    #new_state[position]= np.dot(w_row, state)
    return update(new_state[position], w_row)
 
def dynamics(state, weights, max_iter):
    new_state = state.copy()
    states_list = new_state
    for i in range(max_iter):
   
        if (new_state == state).all() :
            break
 
        new_state = update(state, weights)
        state = new_state
        states_list.append(state)
 
    return states_list
 
def dynamics_async(state, weights, max_iter, convergence_num_iter):
    new_state = state.copy()
    states_list = new_state
    iter=0
 
    for i in range(max_iter):
 
        if (new_state == state).all() :
            iter+=1
            if iter==convergence_num_iter:
                iter=0
                break
 
        new_state = update_async(state, weights)
        state = new_state
        states_list.append(state)
 
    return states_list