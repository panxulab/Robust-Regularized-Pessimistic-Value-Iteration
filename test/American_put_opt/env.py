import numpy as np

class American_put_option():
    def __init__(self, p0, d, seed=1):
        np.random.seed(seed)
        self.lower = 60
        self.upper = 160
        self.p0 = p0
        self.A = [0, 1] # a=0 is not exercising, a=1 is exercising
        self.K = 100
        self.epsilon = 5
        self.c = [1.02, 0.98]
        self.s0 = np.random.uniform(self.K - self.epsilon, self.K + self.epsilon, 1)
        self.d = d
        self.Delta = (self.upper - self.lower) / d
        self.anchor = [80 + x * self.Delta for x in range(d)]

    def reset(self):
        self.S = [self.s0]
        self.A = [] # save the action history
        self.R = [] # save the reward history
        self.h = 0 # reset the step to 0
        self.feature = [] # save the feature trajectory with respect to a=0
        self.current_state = self.s0  # reset the current state to initial state

    def phi(self, s):
        phi = [max(0, (1 - np.abs(s - si) / self.Delta)[0]) for si in self.anchor]    
        #print(phi)
        return np.array(phi)
    
    def add_state(self, s):
        self.S.append(s)
        self.current_state = s

    def update_state(self):
        idx = np.random.choice([0, 1], p = [self.p0, 1 - self.p0])
        sprime = self.current_state * self.c[idx]
        return sprime
    
    def next_state(self, a):
        if a == 1:
            next_state = np.array([99999])
            self.add_state(next_state)
            return next_state
        else:
            next_state = self.update_state()
            # next_state = max(self.lower, next_state)
            # next_state = min(self.upper, next_state)
            self.add_state(next_state)
            return next_state
    
    def generate_reward(self, a):
        if a == 0:
            reward = 0
            self.R.append(reward)
            return reward
        else:
            reward = max(0, (self.K - self.current_state)[0])
            self.R.append(reward)
            return reward
    
    def step(self, a):
        self.A.append(a)
        phi = self.phi(self.current_state)
        self.feature.append(phi)
        self.generate_reward(a)
        self.next_state(a)
        self.h += 1


