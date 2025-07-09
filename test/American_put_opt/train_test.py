import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product
from algorithm import PEVI, DRPVI, DRVI_L, VA_DRPVI, R2PVI_TV, R2PVI_KL, R2PVI_xi2, VA_R2PVI_TV
from data_collection import Offline_Dataset_Collection
from env import American_put_option
from dataclasses import dataclass
import time

def train_agent_once(dataset, algorithm, d, H, beta, lam, rho):
    if algorithm == VA_DRPVI:
        pre_agent = DRPVI(d=d, beta=beta, H=H, lam=lam, dataset=dataset, Rho=rho)
        agent = VA_DRPVI(d=d, beta=beta, H=H, lam=lam, dataset=dataset, Rho=rho, pre_agent=pre_agent)
    elif algorithm == VA_R2PVI_TV:
        pre_agent = R2PVI_TV(d=d, beta=beta, H=H, lam=lam, dataset=dataset, Rho=rho)
        agent = VA_R2PVI_TV(d=d, beta=beta, H=H, lam=lam, dataset=dataset, Rho=rho, pre_agent=pre_agent)
    else:
        agent = algorithm(d=d, beta=beta, H=H, lam=lam, dataset=dataset, Rho=rho)
    agent.update_Q()
    return agent


def test_agent(agent_list, replication, PROB, d, H, T2):
    R_mean = []
    R_std = []
    for p in PROB:
        REWARD = []
        for rep in range(replication):
            reward = 0
            np.random.seed(rep)
            env_test = American_put_option(p, d)
            agent = agent_list[rep]
            for t in range(T2):
                
                env_test.reset()
                
                for h in range(H):
                    current_state = env_test.current_state
                    phi = env_test.phi(current_state)
                    
                    action = agent.get_action(phi, h, current_state)
                    env_test.step(action)    
                reward += np.sum(env_test.R) / T2
            REWARD.append(reward)
            
        R_mean.append(np.mean(REWARD))
        R_std.append(np.std(REWARD))
    return R_mean, R_std    