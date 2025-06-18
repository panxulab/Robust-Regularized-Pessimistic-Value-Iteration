import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product
from algorithm import LSVI_LCB, DR_LSVI_LCB, DR_LSVI_LCB_KL, SRPVI_TV, SRPVI_KL, SRPVI_xi2, VA_DR_LSVI_LCB, VA_SRPVI_TV
from data_collection import Offline_Dataset_Collection
from env import American_put_option
from dataclasses import dataclass
import time
from train_test import train_agent_once, test_agent
import pyrallis
algorithm_dict = {  
                    'LSVI_LCB' :LSVI_LCB, 
                    # distributional robust algorithms
                    'DR_LSVI_LCB': DR_LSVI_LCB, 
                    'DR_LSVI_LCB_KL': DR_LSVI_LCB_KL,
                    'VA_DR_LSVI_LCB': VA_DR_LSVI_LCB,
                    # soft distributional robust algorithms
                    'SRPVI_TV': SRPVI_TV, 
                    'SRPVI_KL': SRPVI_KL, 
                    'SRPVI_xi2': SRPVI_xi2,
                    'VA_SRPVI_TV': VA_SRPVI_TV
                }

@dataclass
class config:    
    N: int = 100
    H: int =  20
    beta: float = 0.1
    lam: float = 1
    rho_list: tuple = (0.025, 0.03)
    p0: float = 0.5
    d: int = 30
    replication: int = 3
    algorithm: str ='DR_LSVI_LCB'
    
@pyrallis.wrap()
def simulate(args: config):

    for rho in args.rho_list:
        print(rho)
        agent_dic = []
        T = 0
        for rep in range(args.replication):
            Offline_Dataset = Offline_Dataset_Collection(args.N, American_put_option(args.p0, args.d, seed=rep))
            start_time = time.time()
            agent = train_agent_once(Offline_Dataset, algorithm_dict[args.algorithm], args.p0, args.d, args.H, args.N, args.beta, args.lam, rho)
            end_time = time.time()
            T += end_time - start_time
            agent_dic.append(agent)
        print('Time:', T / args.replication)
        PROB = [x / 5 for x in range(1,6)]
        T2 = 100
        R_mean, R_std = test_agent(agent_dic, args.replication, PROB, args.d, args.H, T2)
        print(R_mean, R_std)
if __name__ == '__main__':
    simulate()
    

