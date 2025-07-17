import numpy as np

class LinearMDP_train():
    def __init__(self, action_space, delta, xi_norm, seed=1):
        np.random.seed(seed)
        self.state_space = ['x1', 'x2', 'x3', 'x4', 'x5']
        self.action_space = action_space
        self.initial_state = 'x1'
        self.theta = [np.zeros(4),
                      np.array([0,0,0,1]),
                      np.array([0,0,0,1])]
        self.delta = delta
        self.H = 3
        Xi = np.full(len(self.action_space[0]), 1)
        self.Xi = xi_norm * Xi / np.linalg.norm(Xi, 1)

    def reset(self):
        self.S = [self.initial_state]
        self.A = [] # save the action history
        self.R = [] # save the reward history
        self.h = 0 # reset the step to 0
        self.feature = [] # save the feature trajectory
        self.feature_a = [] # save the full feature trajectory
        self.current_state = self.initial_state # reset the current state to initial state

    def phi(self, current_state, A):
        if current_state == 'x1':
            phi = np.array([1 - self.delta - self.Xi @ A, 0, 0, self.delta + self.Xi @ A])
        elif current_state == 'x2':
            phi = np.array([0, 1 - self.delta - self.Xi @ A, 0, self.delta + self.Xi @ A])
        elif current_state == 'x3':
            phi = np.array([0, 0, 1 - self.delta - self.Xi @ A, self.delta + self.Xi @ A])
        elif current_state == 'x4':
            phi = np.array([0, 0, 1, 0])
        else:
            phi = np.array([0, 0, 0, 1])
        return phi

    def add_state(self, s):
        self.S.append(s)
        self.current_state = s

    def update_state(self, phi, h):
        # calculate the transition probability
        if h == 0:
            prob = [phi[0] * (1 - 0.001), 0, phi[0] * 0.001, phi[3]]
        elif h == 1:
            if phi[2] == 1:
                prob = [0, 0, 1, 0]
            elif phi[3] == 1:
                prob = [0, 0, 0, 1]
            else:
                prob = [0, phi[1] * (1 - 0.001), phi[1] * 0.001, phi[3]]
        else:
            prob = [0, 0, phi[2], phi[3]]
        sprime = np.random.choice(range(1,5), size = 1, p = prob)[0]
        return self.state_space[sprime] # return a string
    
    def next_state(self, phi):
        next_state = self.update_state(phi, self.h)
        self.add_state(next_state)
        return next_state
    
    def generate_reward(self, phi):
        reward = np.dot(phi, self.theta[self.h])
        self.R.append(reward)
        return reward
    
    def step(self, a):
        self.A.append(a)
        phi = self.phi(self.current_state, a)
        phi_a = [self.phi(self.current_state, a) for a in self.action_space]
        self.feature.append(phi)
        self.feature_a.append(phi_a)
        self.generate_reward(phi)
        self.next_state(phi)
        self.h += 1
        
class LinearMDP_test():
    
    """Perturbed evironment"""
    
    def __init__(self,  nominal_MDP, q, seed=1):
        np.random.seed(seed)
        self.state_space = nominal_MDP.state_space
        self.action_space = nominal_MDP.action_space
        self.initial_state = 'x1'
        self.theta = nominal_MDP.theta
        self.delta = nominal_MDP.delta
        self.Xi = nominal_MDP.Xi
        self.H = 3
        self.q = q

    def reset(self):
        self.S = [self.initial_state] # save the feature trajectory
        self.A = [] # save the action history
        self.R = [] # save the reward history
        self.h = 0 # reset the step to 0
        self.feature = [] # save the feature trajectory
        self.feature_a = [] # save the full feature trajectory
        self.current_state = self.initial_state # reset the current  state to initial state

    def phi(self, current_state, A):
        if current_state == 'x1':
            phi = np.array([1 - self.delta - self.Xi @ A, 0, 0, self.delta + self.Xi @ A])
        elif current_state == 'x2':
            phi = np.array([0, 1 - self.delta - self.Xi @ A, 0, self.delta + self.Xi @ A])
        elif current_state == 'x3':
            phi = np.array([0, 0, 1 - self.delta - self.Xi @ A, self.delta + self.Xi @ A])
        elif current_state == 'x4':
            phi = np.array([0,0,1,0])
        else:
            phi = np.array([0,0,0,1])
        return phi
    
    def add_state(self, s): 
        self.S.append(s)
        self.current_state = s

    def update_state(self, phi, h):
        # calculate the transition probability  --- perturbed
        if  h == 0:
            prob = [phi[0], 0, self.q * phi[3], (1 - self.q) * phi[3]]
        elif h == 1:
            if phi[2] == 1:
                prob = [0, 0, 1, 0]
            elif phi[3] == 1:
                prob = [0, 0, 0, 1]
            else:
                prob = [0, phi[1] * (1 - 0.001), phi[1] * 0.001, phi[3]]
        else:
            prob =  [0, 0, phi[2], phi[3]]
        sprime = np.random.choice(range(1,5), size = 1, p = prob)[0]
        return self.state_space[sprime] # return a string
    
    def next_state(self, phi):
        next_state = self.update_state(phi, self.h)
        self.add_state(next_state)
        return next_state
    
    def generate_reward(self, phi):
        reward = np.dot(phi, self.theta[self.h])
        self.R.append(reward)
        return reward

    def step(self, a):
        self.A.append(a)
        phi = self.phi(self.current_state, a)
        phi_a = [self.phi(self.current_state, a) for a in self.action_space]
        self.feature.append(phi)
        self.feature_a.append(phi_a)
        self.generate_reward(phi)
        self.next_state(phi)
        self.h += 1


