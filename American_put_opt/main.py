import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from algorithm import PEVI, DRPVI, DRVI_L, VA_DRPVI, R2PVI_TV, R2PVI_KL, R2PVI_xi2, VA_R2PVI_TV
from data_collection import Offline_Dataset_Collection
from env import American_put_option
from dataclasses import dataclass
import time
from train_test import train_agent_once, test_agent
import pyrallis
algorithm_dict = {  
                    'PEVI' :PEVI, 
                    # distributional robust algorithms
                    'DRPVI': DRPVI, 
                    'DRVI_L': DRVI_L,
                    'VA_DRPVI': VA_DRPVI,
                    # soft distributional robust algorithms
                    'R2PVI_TV': R2PVI_TV, 
                    'R2PVI_KL': R2PVI_KL, 
                    'R2PVI_xi2': R2PVI_xi2,
                    'VA_R2PVI_TV': VA_R2PVI_TV
                }

@dataclass
class config:    
    N: int = 100
    H: int =  20
    beta: float = 0.1
    k: float = 1
    rho_list: tuple = (0.025, 0.03)
    p0: float = 0.5
    d: int = 30
    replication: int = 3
    algorithm: str ='PEVI'
    
@pyrallis.wrap()
def simulate(args: config):

    for rho in args.rho_list:
        #print(rho)
        agent_dic = []
        T = 0
        for rep in range(args.replication):
            Offline_Dataset = Offline_Dataset_Collection(args.H, args.N, American_put_option(args.p0, args.d, seed=rep))
            start_time = time.time()
            agent = train_agent_once(Offline_Dataset, algorithm_dict[args.algorithm], args.d, args.H, args.beta, args.k, rho)
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
    

