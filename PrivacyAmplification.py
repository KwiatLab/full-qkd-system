import warnings
import numpy as np
from random import randint

def generate_matrix (rows, columns, alph, seed = randint(0,4294967295)):
    np.random.seed(seed)
    return seed, np.random.randint(alph, size = (rows, columns))

def privacy_amplification (key, HASHED_KEY_LENGTH, alph, seed = None):
    
    ORIGINAL_KEY_LENGTH = len(key)
    hashed_key = np.zeros(HASHED_KEY_LENGTH, dtype=np.int)
    flag = False
    
    if (HASHED_KEY_LENGTH > len(key)):
        warnings.warn("the hashed key length is greater than original key")
    if seed == None:
        flag = True
        (seed, rand_matrix) = generate_matrix(HASHED_KEY_LENGTH, ORIGINAL_KEY_LENGTH, alph)
    else:
        (seed, rand_matrix) = generate_matrix(HASHED_KEY_LENGTH, ORIGINAL_KEY_LENGTH, alph, seed)
#     print "radn",rand_matrix    
    sum = 0
#     print rand_matrix
    for i in range (HASHED_KEY_LENGTH):
        for j in range (ORIGINAL_KEY_LENGTH):
            sum += rand_matrix[i][j] + key[j]
        sum %= alph
        hashed_key[i] = sum
    
    if flag:
        return seed, hashed_key
    return  hashed_key

if __name__ == '__main__':
    print privacy_amplification([0,0,0,0], 3, 7)