import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product

def Offline_Dataset_Collection(sample_size, env, seed=1):
    np.random.seed(seed)
    H = 20
    history = {'S':[], 'A':[], 'R':[], 'phi':[]}    
    epoch = sample_size
    for t in range(epoch):
        env.reset()
        for h in range(H):
            action = 0
            env.step(action)
        # log the trajectory
        history['S'].append(env.S)
        history['A'].append(env.A)
        history['R'].append(env.R)
        history['phi'].append(env.feature)
        #history['k'] += 1
    return history
