import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product
from algorithm import PEVI, DRPVI, DRVI_L, VA_DRPVI, R2PVI_TV, R2PVI_KL, R2PVI_xi2, VA_R2PVI_TV
from data_collection import Offline_Dataset_Collection
from env import LinearMDP_test, LinearMDP_train
from dataclasses import dataclass
import time

def train_agent_once(dataset, algorithm, action_space, beta, H, lam, Rho, theta, fail_state):
    if algorithm == VA_DRPVI:
        pre_agent = DRPVI(A=action_space, beta=beta, H=H, lam=lam, dataset=dataset, Rho=Rho, theta=theta, fail_state=fail_state)
        agent = VA_DRPVI(pre_agent=pre_agent, A=action_space, beta=beta, H=H, lam=lam, dataset=dataset, Rho=Rho, theta=theta, fail_state=fail_state)
    elif algorithm == VA_R2PVI_TV:
        pre_agent = R2PVI_TV(A=action_space, beta=beta, H=H, lam=lam, dataset=dataset, Rho=Rho, theta=theta, fail_state=fail_state)
        agent = VA_R2PVI_TV(pre_agent=pre_agent, A=action_space, beta=beta, H=H, lam=lam, dataset=dataset, Rho=Rho, theta=theta, fail_state=fail_state)
    else:
        agent = algorithm(A=action_space, beta=beta, H=H, lam=lam, dataset=dataset, Rho=Rho, theta=theta, fail_state=fail_state)
    agent.update_Q()
    return agent


def test_agent(agent_list, env, replication, Perturbation, H, T2):
    R_mean = []
    R_std = []
    for q in Perturbation:
        REWARD = []
        for rep in range(replication):
            reward = 0
            env_test = LinearMDP_test(env, q=q, seed=rep)
            agent = agent_list[rep]
            for t in range(T2):
                
                env_test.reset()
                
                for h in range(H):
                    current_state = env_test.current_state
                    phi = [env_test.phi(current_state, a) for a in env.action_space]
                    
                    action = agent.get_action(phi, h)
                    env_test.step(action)    
                reward += np.sum(env_test.R) / T2
            REWARD.append(reward)
            
        R_mean.append(np.mean(REWARD))
        R_std.append(np.std(REWARD))
    return R_mean, R_std    