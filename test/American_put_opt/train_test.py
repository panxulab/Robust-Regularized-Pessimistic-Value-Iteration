import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product
from algorithm import LSVI_LCB, DR_LSVI_LCB, SRPVI_TV, VA_DR_LSVI_LCB, VA_SRPVI_TV
from data_collection import Offline_Dataset_Collection
from env import American_put_option
from dataclasses import dataclass
import pyrallis
import time

def train_agent_once(dataset, algorithm, p0, d, H, T1, beta, lam, rho):
    if algorithm == VA_DR_LSVI_LCB:
        pre_agent = DR_LSVI_LCB(d=d, beta=beta, H=H, lam=lam, dataset=dataset, Rho=rho)
        agent = VA_DR_LSVI_LCB(d=d, beta=beta, H=H, lam=lam, dataset=dataset, Rho=rho, pre_agent=pre_agent)
    elif algorithm == VA_SRPVI_TV:
        pre_agent = SRPVI_TV(d=d, beta=beta, H=H, lam=lam, dataset=dataset, Rho=rho)
        agent = VA_SRPVI_TV(d=d, beta=beta, H=H, lam=lam, dataset=dataset, Rho=rho, pre_agent=pre_agent)
    else:
        agent = algorithm(d=d, beta=beta, H=H, lam=lam, dataset=dataset, Rho=rho)
    agent.update_Q()
    return agent


def test_agent(agent_list, replication, PROB, d, H, T2):
    R_mean = []
    R_std = []
    for p in PROB:
        REWARD = []
        REWARD_DR = []
        for rep in range(replication):
            reward = 0
            reward_DR = 0
            np.random.seed(rep)
            env_test = American_put_option(p, d)
            agent = agent_list[rep]
            for t in range(T2):
                
                env_test.reset()
                
                for h in range(H):
                    current_state = env_test.current_state
                    #print(current_state)
                    phi = env_test.phi(current_state)
                    
                    action = agent.get_action(phi, h, current_state)
                    #action = 1
                    #print(action)
                    env_test.step(action)    
                #print(env_test_DR.R)
                #print(env_test.R)
                reward += np.sum(env_test.R) / T2
                #print(reward_DR)
            #import IPython; IPython.embed()
            #print(reward)
            REWARD.append(reward)
            REWARD_DR.append(reward_DR)
            
        R_mean.append(np.mean(REWARD))
        R_std.append(np.std(REWARD))
    return R_mean, R_std    