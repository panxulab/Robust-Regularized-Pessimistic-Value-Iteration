import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from algorithm import PEVI, DRPVI, DRVI_L, VA_DRPVI, R2PVI_TV, R2PVI_KL, R2PVI_xi2, VA_R2PVI_TV
from data_collection import Offline_Dataset_Collection
from env import LinearMDP_train, LinearMDP_test
from dataclasses import dataclass
from itertools import product
from train_test import train_agent_once, test_agent
import pyrallis
from typing import Tuple, Dict, List
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
    T1: int = 100                  # Number of training episodes
    H: int = 3                     # Horizon length
    beta: float = 1                # Regularization parameter
    k: float = 0.1                 # k parameter
    delta: float = 0.3             # Delta parameter
    xi_norm: float = 0.2           # xi norm parameter
    rho: float = 0.1               # Rho parameter
    lam: float = 0.1               # Lambda parameter
    fail_state: str = 'x4'         # Fail state identifier
    replication: int = 20          # Number of replications
    T2: int = 100                  # Number of test episodes
    perturbation_steps: int = 11   # Number of perturbation steps (0.0 to 1.0)
    algorithms: tuple = ('PEVI', 'R2PVI_KL', 'R2PVI_xi2', 'R2PVI_TV')  # Algorithms to run
    
@pyrallis.wrap()
def simulate(args: config) -> Dict[str, Dict[str, List[float]]]:

    # Initialize action space and results
    actions = list(product([-1, 1], repeat=4))
    action_space = [np.array(action) for action in actions]
    results = {}
    
    # Initialize environment and parameters
    env = LinearMDP_train(action_space, args.delta, args.xi_norm)
    Rho = [[0, 0, 0, args.rho], [0, 0, 0, 0]]
    perturbations = [x / (args.perturbation_steps - 1) for x in range(args.perturbation_steps)]
    
    # Train and test each selected algorithm
    for algo_name in args.algorithms:
        if algo_name not in algorithm_dict:
            raise ValueError(f"Unknown algorithm: {algo_name}")
            
        print(f"Training {algo_name} agents...")
        agent_list = []
        algorithm = algorithm_dict[algo_name]
        
        # Train agents for all replications
        for rep in range(args.replication):
            dataset = Offline_Dataset_Collection(args.T1, env, seed=rep)
            if 'R2PVI' in algo_name:
                agent = train_agent_once(
                    dataset=dataset,
                    algorithm=algorithm,
                    action_space=action_space,
                    beta=args.beta,
                    H=args.H,
                    lam=args.k,
                    Rho=args.lam,
                    theta=env.theta,
                    fail_state=args.fail_state
                )
            else:
                agent = train_agent_once(
                    dataset=dataset,
                    algorithm=algorithm,
                    action_space=action_space,
                    beta=args.beta,
                    H=args.H,
                    lam=args.k,
                    Rho=Rho,
                    theta=env.theta,
                    fail_state=args.fail_state
                )
            agent_list.append(agent)
        
        # Test the trained agents
        print(f"Testing {algo_name} agents...")
        R_mean, R_std = test_agent(
            agent_list=agent_list,
            env=env,
            replication=args.replication,
            Perturbation=perturbations,
            H=args.H,
            T2=args.T2
        )

        results[algo_name] = {
            'mean': R_mean,
            'std': R_std
        }

    plt.plot(perturbations, results['PEVI']['mean'], label = 'PEVI', marker = 'D', color = 'gray')    
    plt.plot(perturbations, results['R2PVI_TV']['mean'], label = 'R2PVI-TV', marker = 'X')
    plt.plot(perturbations, results['R2PVI_KL']['mean'], label = 'R2PVI-KL', marker = 'D')
    plt.plot(perturbations, results['R2PVI_xi2']['mean'], label = r'R2PVI-$\chi^2$', marker = 'o', color = 'lightblue')
    plt.legend(fontsize=16)
    plt.grid(alpha=0.3)
    plt.xlabel('Perturbation', size=16)
    plt.ylabel('Average reward', size=16)
    plt.savefig(f'result__{args.rho}.pdf', dpi=1000, bbox_inches='tight', pad_inches=0.0)
    
    return results

if __name__ == '__main__':
    # Run simulation with configurable parameters
    results = simulate()
    
    # Example of accessing results
    for algo_name, algo_results in results.items():
        print(f"\n{algo_name} Results:")
        print(f"Means: {algo_results['mean']}")
        print(f"Std devs: {algo_results['std']}")

