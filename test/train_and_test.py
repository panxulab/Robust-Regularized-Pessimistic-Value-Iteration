from algorithm import LSVI_LCB, DR_LSVI_LCB, VA_DR_LSVI_LCB, SRPVI_TV, SRPVI_KL, SRPVI_xi2
import numpy as np
from env import LinearMDP_test, LinearMDP_train
from dataset_collection import Offline_Dataset_Collection
from itertools import product
import matplotlib.pyplot as plt

def train_once(dataset, action_space, beta, H, lam, fail_state):
    agent = LSVI_LCB(A=action_space, beta=beta, H=H, lam=lam, dataset=dataset, fail_state=fail_state)
    agent.update_Q()
    return agent

def train_once_DR(dataset, action_space, beta, H, lam, Rho, theta, fail_state):
    agent = DR_LSVI_LCB(A=action_space, beta=beta, H=H, lam=lam, dataset=dataset, Rho=Rho, theta=theta, fail_state=fail_state)
    agent.update_Q()
    return agent

def train_once_DR_VA(dataset, action_space, beta, H, lam, Rho, theta, pre_agent, fail_state):
    agent = VA_DR_LSVI_LCB(pre_agent=pre_agent, A=action_space, beta=beta, H=H, lam=lam, dataset=dataset, Rho=Rho, theta=theta, fail_state=fail_state)
    agent.update_Q()
    return agent

def train_once_SRPVI_TV(dataset, action_space, beta, H, lam, Rho, theta, fail_state):
    agent = SRPVI_TV(A=action_space, beta=beta, H=H, lam=lam, dataset=dataset, Rho=Rho, theta=theta, fail_state=fail_state)
    agent.update_Q()
    return agent

def train_once_SRPVI_KL(dataset, action_space, beta, H, lam, Rho, theta, fail_state):
    agent = SRPVI_KL(A=action_space, beta=beta, H=H, lam=lam, dataset=dataset, Rho=Rho, theta=theta, fail_state=fail_state)
    agent.update_Q()
    return agent

def train_once_SRPVI_xi2(dataset, action_space, beta, H, lam, Rho, theta, fail_state):
    agent = SRPVI_xi2(A=action_space, beta=beta, H=H, lam=lam, dataset=dataset, Rho=Rho, theta=theta, fail_state=fail_state)
    agent.update_Q()
    return agent

if __name__ == '__main__':
    T1 = 100
    H = 3
    rho = 0.3
    beta = 1
    lam = 0.1
    actions = list(product([-1, 1], repeat=4))
    action_space = [np.array(action) for action in actions]
    delta = 0.3
    xi_norm = 0.1
    Rho = [[0,0,0,rho], [0,0,0,rho]]
    fail_state = 'x4'
    replication = 20
    agent_dic = {}
    DR_agent_dic = {}
    VA_DR_agent_dic = {}
    SRPVI_TV_agent_dic = {}
    SRPVI_KL_agent_dic = {}
    SRPVI_xi2_agent_dic = {}
    env = LinearMDP_train(action_space, delta, xi_norm)

    for rep in range(replication):
        Offline_Dataset = Offline_Dataset_Collection(T1, env, seed=rep)
        agent = train_once(dataset=Offline_Dataset, action_space=action_space, beta=beta, H=H, lam=lam, fail_state=fail_state)
        DR_agent = train_once_DR(dataset=Offline_Dataset, action_space=action_space, beta=beta, H=H, lam=lam, Rho=Rho, theta=env.theta, fail_state=fail_state)
        VA_DR_agent = train_once_DR_VA(pre_agent=DR_agent, dataset=Offline_Dataset, action_space=action_space, beta=beta, H=H, lam=lam, Rho=Rho, theta=env.theta, fail_state=fail_state)
        SRPVI_TV_agent = train_once_SRPVI_TV(dataset=Offline_Dataset, action_space=action_space, beta=beta, H=H, lam=lam, Rho=Rho, theta=env.theta, fail_state=fail_state)
        SRPVI_KL_agent = train_once_SRPVI_KL(dataset=Offline_Dataset, action_space=action_space, beta=beta, H=H, lam=lam, Rho=Rho, theta=env.theta, fail_state=fail_state)
        SRPVI_xi2_agent = train_once_SRPVI_xi2(dataset=Offline_Dataset, action_space=action_space, beta=beta, H=H, lam=lam, Rho=Rho, theta=env.theta, fail_state=fail_state)
        agent_dic[str(rep)] = agent
        
        DR_agent_dic[str(rep)] = DR_agent
        VA_DR_agent_dic[str(rep)] = VA_DR_agent
        SRPVI_TV_agent_dic[str(rep)] = SRPVI_TV_agent
        SRPVI_KL_agent_dic[str(rep)] = SRPVI_KL_agent
        SRPVI_xi2_agent_dic[str(rep)] = SRPVI_xi2_agent
    
    Perturbation = [x / 20 for x in range(21)]
    T2 = 100
    R_LSVI_LCB = []
    R_DR_LSVI_LCB = []
    R_VA_DR_LSVI_LCB = []
    R_SRPVI_TV = []
    R_SRPVI_KL = []
    R_SRPVI_xi2 = []
    env = LinearMDP_train(action_space, delta, xi_norm)
    for q in Perturbation:
        REWARD = 0
        REWARD_DR = 0
        REWARD_DR_VA = 0
        REWARD_SRPVI_TV = 0
        REWARD_SRPVI_KL = 0
        REWARD_SRPVI_xi2 = 0
        for rep in range(replication):
            reward = 0
            reward_DR = 0
            reward_DR_VA = 0
            reward_SRPVI_TV = 0
            reward_SRPVI_KL = 0
            reward_SRPVI_xi2 = 0
            env_test = LinearMDP_test(env, q=q, seed=rep)
            env_test_DR = LinearMDP_test(env, q=q, seed=rep)
            env_test_DR_VA = LinearMDP_test(env, q=q, seed=rep)
            env_test_SRPVI_TV = LinearMDP_test(env, q=q, seed=rep)
            env_test_SRPVI_KL = LinearMDP_test(env, q=q, seed=rep)
            env_test_SRPVI_xi2 = LinearMDP_test(env, q=q, seed=rep)
            agent = agent_dic[str(rep)]
            DR_agent = DR_agent_dic[str(rep)]
            VA_DR_agent = VA_DR_agent_dic[str(rep)]
            SRPVI_TV_agent = SRPVI_TV_agent_dic[str(rep)]
            SRPVI_KL_agent = SRPVI_KL_agent_dic[str(rep)]
            SRPVI_xi2_agent = SRPVI_xi2_agent_dic[str(rep)]
            for t in range(T2):
                env_test.reset()
                env_test_DR.reset()
                env_test_DR_VA.reset()
                env_test_SRPVI_TV.reset()
                env_test_SRPVI_KL.reset()
                env_test_SRPVI_xi2.reset()
                for h in range(H):
                # PEVI
                    current_state = env_test.current_state
                    phi_a = [env_test.phi(current_state, a) for a in action_space]
                    action = agent.get_action(phi_a, h)
                    env_test.step(action)
                # DRPVI
                    current_state_DR = env_test_DR.current_state
                    phi_DR_a = [env_test_DR.phi(current_state_DR, a) for a in action_space]
                    action_DR = DR_agent.get_action(phi_DR_a, h)
                    env_test_DR.step(action_DR)
                
                # VA-DRPVI
                    current_state_DR_VA = env_test_DR_VA.current_state
                    phi_DR_VA_a = [env_test_DR_VA.phi(current_state_DR_VA, a) for a in action_space]
                    action_DR_VA = VA_DR_agent.get_action(phi_DR_VA_a, h)
                    env_test_DR_VA.step(action_DR_VA)
                
                # SRPVI-TV
                    current_state_SDR = env_test_SRPVI_TV.current_state
                    phi_SDR_a = [env_test_SRPVI_TV.phi(current_state_SDR, a) for a in action_space]
                    action_SDR = SRPVI_TV_agent.get_action(phi_SDR_a, h)
                    env_test_SRPVI_TV.step(action_SDR)

                # SRPVI-KL
                    current_state_SDR = env_test_SRPVI_KL.current_state
                    phi_SDR_a = [env_test_SRPVI_KL.phi(current_state_SDR, a) for a in action_space]
                    action_SDR = SRPVI_KL_agent.get_action(phi_SDR_a, h)
                    env_test_SRPVI_KL.step(action_SDR)
                 
                # SRPVI-xi2
                    current_state_SDR = env_test_SRPVI_xi2.current_state
                    phi_SDR_a = [env_test_SRPVI_xi2.phi(current_state_SDR, a) for a in action_space]
                    action_SDR = SRPVI_xi2_agent.get_action(phi_SDR_a, h)
                    env_test_SRPVI_xi2.step(action_SDR)
                
                reward += np.sum(env_test.R) / T2
                reward_DR += np.sum(env_test_DR.R) / T2   
                reward_DR_VA += np.sum(env_test_DR_VA.R) / T2 
                reward_SRPVI_TV += np.sum(env_test_SRPVI_TV.R) / T2
                reward_SRPVI_KL += np.sum(env_test_SRPVI_KL.R) / T2
                reward_SRPVI_xi2 += np.sum(env_test_SRPVI_xi2.R) / T2
            
            REWARD += reward / replication
            REWARD_DR += reward_DR / replication 
            REWARD_DR_VA += reward_DR_VA / replication 
            REWARD_SRPVI_TV += reward_SRPVI_TV / replication
            REWARD_SRPVI_KL += reward_SRPVI_KL / replication
            REWARD_SRPVI_xi2 += reward_SRPVI_xi2 / replication
        
        R_LSVI_LCB.append(REWARD)
        R_DR_LSVI_LCB.append(REWARD_DR)
        R_VA_DR_LSVI_LCB.append(REWARD_DR_VA)
        R_SRPVI_TV.append(REWARD_SRPVI_TV)
        R_SRPVI_KL.append(REWARD_SRPVI_KL)
        R_SRPVI_xi2.append(REWARD_SRPVI_xi2)
    
    plt.plot(Perturbation, R_LSVI_LCB, label = 'PEVI')
    plt.plot(Perturbation, R_DR_LSVI_LCB, label = 'DRPVI')
    plt.plot(Perturbation, R_VA_DR_LSVI_LCB, label = 'VA-DRPVI')
    plt.plot(Perturbation, R_SRPVI_TV, label = 'SRPVI-TV')
    plt.plot(Perturbation, R_SRPVI_KL, label = 'SRPVI-KL')
    plt.plot(Perturbation, R_SRPVI_xi2, label = 'SRPVI-xi2')

    plt.legend(fontsize=16)
    plt.xlabel('Perturbation', size=16)
    plt.ylabel('Average reward', size=16)
    plt.savefig(f'robustness_{delta}_{xi_norm}_{rho}.pdf', dpi=1000, bbox_inches='tight', pad_inches=0.0)