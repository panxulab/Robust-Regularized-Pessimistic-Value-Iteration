import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product

def Offline_Dataset_Collection(sample_size, env, seed=1):
    np.random.seed(seed)
    history = {'k': 0, 'phi':[], 'r':[], 'state':[], 'phi_a':[]}    
    epoch = sample_size
    for t in range(epoch):
        env.reset()
        for h in range(env.H):
            random_action_index = np.random.choice(range(0,len(env.action_space)), size = 1)[0]
            action = env.action_space[random_action_index]
            env.step(action)
        # log the trajectory
        history['phi_a'].append(env.feature_a)
        history['phi'].append(env.feature)
        history['r'].append(env.R)
        history['state'].append(env.S)
        history['k'] += 1
    return history
