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
   
    #new_state = np.zeros(np.size(weights, 1))
    #new_state = np.empty(np.size(weights, 1))
   
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
    new_state = update(state, weights)
    states_list = new_state
    for i in range(max_iter):

        #new_state = update(state, weights)

        if (new_state == state).all() :
            break

        new_state = update(state, weights)
        state = new_state
        #states_list.append(state)
        states_list = np.insert(states_list, np.shape(states_list)[0], state)
 
    return states_list
 
def dynamics_async(state, weights, max_iter, convergence_num_iter):
    new_state = update_async(state, weights)
    states_list = new_state
    iter=0
    """"""
    for i in range(max_iter):
 
        if (new_state == state).all() :
            iter+=1
            if iter==convergence_num_iter:
                iter=0
                break
 
        new_state = update_async(state, weights)
        state = new_state
        #states_list.append(state)
        states_list = np.insert(states_list, np.shape(states_list)[0], state)
 
    return states_list

def storkey_weights(patterns):
    N = np.size(patterns[0])
    M = np.shape(patterns)[0]
    W = np.zeros((N,N))
    W_previous = np.zeros((N,N))
    # maybe don't need the mu and you can do with when you do the steps
    H = np.zeros((N,N))
    for mu in range(M):
        print('mu =', mu)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    """""
                    if k != i and k != j:
                        H[i][j] += W_previous[i][k]*patterns[mu][k]
                    """
                    H = np.matmul(W_previous, patterns[mu].T)
                W[i][j] = W_previous[i][j] + (1/N)*(patterns[mu][i]*patterns[mu][j] - patterns[mu][i]*H[j][i] - patterns[mu][j]*H[i][j])
        W_previous = W
    return W


