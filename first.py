import numpy as np
N = 50
M = 3
p = np.random.choice([-1,1], (M,N))

W = (1/M)*(np.matmul(np.transpose(p),p)) - np.identity(N)

answer = True
answer2 = True
for i in range(N):
    for j in range(N):
        if W[i][j] < -1 or W[i][j] > 1:
            answer2 = False
if (W != W.T).all():
    answer = False

print('The matrix is symmetric: ', answer)
print('The elements of the matrix are between -1 and 1: ', answer2)
 
line = np.random.randint(0,2)
position_1 = np.random.randint(0,N-1)
position_2 = np.random.randint(0,N-1)
position_3 = np.random.randint(0,N-1)

while position_1 == position_2 or position_1 == position_3 or position_2 == position_3:
    position_2 = np.random.randint(0,N-1)
    position_3 = np.random.randint(0,N-1)

value_1 = np.random.choice([-1,1])
value_2 = np.random.choice([-1,1])
value_3 = np.random.choice([-1,1])

p_1 = p[line].copy()
p_1[position_1] = value_1
p_1[position_2] = value_2
p_1[position_3] = value_3

p_2 = np.zeros((1,N))
for i in range(20):
    
    if (p_1 == p[line]).all() :
        break
        
    p_2 = np.dot(W, p_1.T)
    for i in range(np.size(p_1)):
        if p_2[i] >= 0:
            p_2[i] = 1
        if p_2[i] < 0:
            p_2[i] = -1
    p_1 = p_2

#to verify convergence:
if (p_1 == p[line]).all():
    print('same at end?', True)
else:
    print('same at end?', False)

# If we change more values, we will have to do more iterations
# to have convergence
# If we have less changes, we need less iterations
