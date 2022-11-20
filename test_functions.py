import numpy as np
import functions as f
#import main 
#import pytest

def test_generate_patterns():
    '''
    patterns = np.array([-1,-1,1],
                        [1,1,1])
    '''
    patterns = f.generate_patterns(2,3)
    assert np.shape(patterns) == (2,3)
    print('tested 1')
    '''
    for i in range(2):
        for j in range(3):
            assert f.generate_patterns(2,3)[i][j] == -1 or 1
            print('tested 2')
            '''
    
    patterns[1][1] = 3
    print(patterns)
    for i in range(2):
        #assert np.all(patterns[i]) == -1 or 0
        assert (patterns[i]).any() == -1 or 1