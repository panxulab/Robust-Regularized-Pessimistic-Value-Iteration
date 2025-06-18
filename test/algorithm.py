import numpy as np
from scipy.optimize import minimize

class meta_algorithm():
    def __init__(self, A, beta, H, lam, dataset, fail_state=None):
        self.lam = lam
        self.H = H
        self.action_space = A
        self.beta = beta
        self.w = [np.zeros(4) for _ in range(self.H)]
        self.fail_state = fail_state
        self.Lambda = [self.lam * np.diag(np.ones(4)) for _ in range(self.H)]
        self.dataset = dataset
    
    def get_action(self, phi_a, h):
        Q_h = [self.get_Q_func(phi_a[idx], h) for idx in range (len(self.action_space))]
        return self.action_space[np.argmax(Q_h)]
    
    def get_Q_func(self, phi, h, s_f=False):
        if s_f == True:
            return 0
        else:
            Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
            Q_h = np.min([(self.w[h] @ phi - self.beta * np.sqrt(phi @ Lambda_h_inverse @ phi)), self.H])
            return Q_h

    def get_nu_h(self, h, rho, variance):
        pass    
                    
    def update_Q(self):
        # Backward induction
        self.w = [None for _ in range(self.H)] # innitialize weights w
        for h in range(self.H-1, -1, -1):
            # calculate Lambda_h
            for n in range(self.dataset['k']):
                feature_temp = self.dataset['phi'][n][h]
                self.Lambda[h] += np.outer(feature_temp, feature_temp)
            # update w_h
            w_h = np.zeros(4)
            nu_h = np.zeros(4)
            if h == self.H - 1:
                w_h = self.theta[h]
            else:
                nu_h = self.get_nu_h(h, rho=self.Rho[h])
                w_h = self.theta[h] + nu_h
            self.w[h] = w_h 
            
class LSVI_LCB():
    def __init__(self, A, beta, H, lam, dataset, fail_state=None):
        self.lam = lam
        self.H = H
        self.action_space = A
        self.beta = beta
        self.w = [np.zeros(4) for _ in range(self.H)]
        self.fail_state = fail_state
        self.Lambda = [self.lam * np.diag(np.ones(4)) for _ in range(self.H)]
        self.dataset = dataset
    
    def get_action(self, phi_a, h):
        Q_h = [self.get_Q_func(phi_a[idx], h) for idx in range (len(self.action_space))]
        return self.action_space[np.argmax(Q_h)]
    
    def get_Q_func(self, phi, h, s_f=False):
        if s_f == True:
            return 0
        else:
            Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
            Q_h = np.min([(self.w[h] @ phi - self.beta * np.sqrt(phi @ Lambda_h_inverse @ phi)), self.H])
            return Q_h
    
    def update_Q(self):
        # Backward induction
        self.w = [None for _ in range(self.H)] #initialize weights w
        for h in range(self.H-1, -1, -1):
            # calculate Lambda_h
            for n in range(self.dataset['k']):
                feature_temp = self.dataset['phi'][n][h]
                self.Lambda[h] += np.outer(feature_temp, feature_temp)
            #  calculate w_h
            Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
            w_h = np.zeros(self.d)
            if h == self.H - 1:
                for tau in range(self.dataset['k']):
                    phi_tau_h = self.dataset['phi'][tau][h]
                    r_tau_h = self.dataset['r'][tau][h]
                    w_h += Lambda_h_inverse @ (phi_tau_h * r_tau_h)
            else:
                for tau in range(self.dataset['k']):
                    phi_tau_h = self.dataset['phi'][tau][h]
                    phi_tau_h_plus_one = self.dataset['phi_a'][tau][h+1]
                    r_tau_h = self.dataset['r'][tau][h]
                    s_f_h_plus_one = (self.dataset['state'][tau][h+1] == self.fail_state)
                    Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one[idx], h + 1, s_f_h_plus_one)
                                    for idx in range(len(self.action_space))]
                    V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
                    w_h += Lambda_h_inverse @ (phi_tau_h * (r_tau_h + V_tau_h_plus_one))
            self.w[h] = w_h

class DR_LSVI_LCB():
    def __init__(self, A, beta, H, lam, dataset, Rho, theta, fail_state=None):
        self.lam = lam
        self.H = H
        self.action_space = A
        self.beta = beta
        self.w = [np.zeros(4) for _ in range(self.H)]
        self.Lambda = [self.lam * np.diag(np.ones(4)) for _ in range(self.H)]
        self.Rho = Rho
        self.theta = theta
        self.fail_state = fail_state
        self.dataset = dataset

    def get_action(self, phi_a, h):
        Q_h = [self.get_Q_func(phi_a[idx], h) for idx in range(len(self.action_space))]
        return self.action_space[np.argmax(Q_h)]

    def get_Q_func(self, phi, h, s_f=False):
        if s_f == True:
            return 0
        else:
            Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
            penalty = self.beta * np.sqrt(phi @ np.diag(np.diagonal(Lambda_h_inverse)) @ phi)
            Q_h = np.min([(self.w[h] @ phi - penalty), self.H - h])
            return Q_h

    def get_nu_h(self, h, rho):
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        nu_h = np.zeros(4)
        Phi_h = np.zeros((0,4))
        V_h_plus_one = np.zeros(0)
        for tau in range(self.dataset['k']):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h = np.vstack((Phi_h, phi_tau_h))
            phi_tau_h_plus_one = self.dataset['phi_a'][tau][h+1]
            s_f_h_plus_one = (self.dataset['state'][tau][h+1] == self.fail_state)
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one[idx], h+1, s_f_h_plus_one)
                                for idx in range(len(self.action_space))]
            V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
            V_h_plus_one = np.hstack((V_h_plus_one, V_tau_h_plus_one))
        for i in range(4):
            def z_alpha_i(alpha):
                # compact formular for z
                z = Lambda_h_inverse @ Phi_h.T @ np.minimum(V_h_plus_one, alpha)
                return -z[i] + rho[i] * alpha
            result = minimize(z_alpha_i, self.H/2, method='Nelder-Mead', bounds=[(0, self.H)])
            nu_h[i] = - result.fun
        return nu_h

    def update_Q(self):
        # Backward induction
        self.w = [None for _ in range(self.H)] # innitialize weights w
        for h in range(self.H-1, -1, -1):
            # calculate Lambda_h
            for n in range(self.dataset['k']):
                feature_temp = self.dataset['phi'][n][h]
                self.Lambda[h] += np.outer(feature_temp, feature_temp)
            # update w_h
            w_h = np.zeros(4)
            nu_h = np.zeros(4)
            if h == self.H - 1:
                w_h = self.theta[h]
            else:
                nu_h = self.get_nu_h(h, rho=self.Rho[h])
                w_h = self.theta[h] + nu_h
            self.w[h] = w_h

class VA_DR_LSVI_LCB():
    def __init__(self, pre_agent, A, beta, H, lam, dataset, Rho, theta, fail_state=None):
        self.lam = lam
        self.H = H
        self.action_space = A
        self.beta = beta
        self.w = [np.zeros(4) for _ in range(self.H)]
        self.Lambda = [self.lam * np.diag(np.ones(4)) for _ in range(self.H)]
        self.Rho = Rho
        self.theta = theta
        self.fail_state = fail_state
        self.dataset = dataset
        self.pre_agent = pre_agent
        self.variance = {}

    def get_variance_coefficient(self, h):
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        z1_h = np.zeros(4)
        z2_h = np.zeros(4)
        Phi_h = np.zeros((0,4))
        V_h_plus_one = np.zeros(0)
        for tau in range(self.dataset['k']):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h = np.vstack((Phi_h, phi_tau_h))
            phi_tau_h_plus_one = self.dataset['phi_a'][tau][h+1]
            s_f_h_plus_one = (self.dataset['state'][tau][h+1] == self.fail_state)
            Q_tau_h_plus_one = [self.pre_agent.get_Q_func(phi_tau_h_plus_one[idx], h+1, s_f_h_plus_one)
                                for idx in range(len(self.action_space))]
            V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
            V_h_plus_one = np.hstack((V_h_plus_one, V_tau_h_plus_one))
        
        z1_h = Lambda_h_inverse @ Phi_h.T @ V_h_plus_one
        z2_h = Lambda_h_inverse @ Phi_h.T @ V_h_plus_one**2
        return z1_h, z2_h
    
    def estimated_variance(self, phi, z1_h, z2_h):
        second_order_term = np.min([np.max([0, np.dot(phi, z2_h)]), self.H**2])
        first_order_term = np.min([np.max([0, np.dot(phi, z1_h)]), self.H])
        sigma_square = np.max([1, second_order_term - first_order_term**2])
        return sigma_square
    
    def get_action(self, phi_a, h):
        Q_h = [self.get_Q_func(phi_a[idx], h) for idx in range(len(self.action_space))]
        return self.action_space[np.argmax(Q_h)]
    
    def get_Q_func(self, phi, h, s_f=False):
        if s_f == True:
            return 0
        else:
            Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
            penalty = self.beta * np.sqrt(phi @ np.diag(np.diagonal(Lambda_h_inverse)) @ phi)
            Q_h = np.min([(self.w[h] @ phi - penalty), self.H - h])
            return Q_h

    def get_nu_h(self, h, rho, variance):
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        nu_h = np.zeros(4)
        Phi_h = np.zeros((0, 4))
        V_h_plus_one = np.zeros(0)
        for tau in range(self.dataset['k']):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h = np.vstack((Phi_h, phi_tau_h))
            phi_tau_h_plus_one = self.dataset['phi_a'][tau][h+1]
            s_f_h_plus_one = (self.dataset['state'][tau][h+1] == self.fail_state)
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one[idx], h+1, s_f_h_plus_one)
                                for idx in range(len(self.action_space))]
            V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
            V_h_plus_one = np.hstack((V_h_plus_one, V_tau_h_plus_one))
        for i in range(4):
            def z_alpha_i(alpha):
                # compact formular for z
                z = Lambda_h_inverse @ Phi_h.T @ (np.minimum(V_h_plus_one, alpha) / variance)
                return -z[i] + rho[i] * alpha
            result = minimize(z_alpha_i, self.H/2, method='Nelder-Mead', bounds=[(0, self.H)])
            nu_h[i] = - result.fun
            return nu_h
        
    def update_Q(self):
        # Backward induction
        self.w = [None for _ in range(self.H)] # initialize weights w
        for h in range(self.H-1, -1, -1):
            # calculate Lambda_h
            if h == self.H-1:
                for n in range(self.dataset['k']):
                    feature_temp = self.dataset['phi'][n][h]
                    self.Lambda[h] += np.outer(feature_temp, feature_temp)
            else:
                z1_h, z2_h = self.get_variance_coefficient(h)
                self.variance[str(h)] = np.zeros(self.dataset['k'])
                for n in range(self.dataset['k']):
                    feature_temp = self.dataset['phi'][n][h]
                    variance_temp = self.estimated_variance(feature_temp, z1_h, z2_h)
                    self.variance[str(h)][n] = variance_temp
                    self.Lambda[h] += np.outer(feature_temp, feature_temp) / variance_temp
            # update w_h
            w_h = np.zeros(4)
            nu_h = np.zeros(4)
            if h == self.H - 1:
                w_h = self.theta[h]
            else:
                variance = self.variance[str(h)]
                nu_h = self.get_nu_h(h, rho = self.Rho[h], variance = variance)
                w_h = self.theta[h] + nu_h
            self.w[h] = w_h

class SRPVI_TV():
    def __init__(self, A, beta, H, lam, dataset, Rho, theta, rho=0.2, fail_state=None):
        self.lam = lam
        self.H = H
        self.action_space = A
        self.beta = beta
        self.w = [np.zeros(4) for _ in range(self.H)]
        self.Lambda = [self.lam * np.diag(np.ones(4)) for _ in range(self.H)]
        self.Rho = Rho
        self.theta = theta
        self.fail_state = fail_state
        self.dataset = dataset
        self.rho = rho
        
    def get_action(self, phi_a, h):
        Q_h = [self.get_Q_func(phi_a[idx], h) for idx in range(len(self.action_space))]
        return self.action_space[np.argmax(Q_h)]

    def get_Q_func(self, phi, h, s_f=False):
        if s_f == True:
            return 0
        else:
            Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
            penalty = self.beta * np.sqrt(phi @ np.diag(np.diagonal(Lambda_h_inverse)) @ phi)
            Q_h = np.min([(self.w[h] @ phi - penalty), self.H - h])
            return Q_h

    def get_nu_h(self, h, rho):
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        nu_h = np.zeros(4)
        Phi_h = np.zeros((0,4))
        V_h_plus_one = np.zeros(0)
        
        for tau in range(self.dataset['k']):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h = np.vstack((Phi_h, phi_tau_h))
            phi_tau_h_plus_one = self.dataset['phi_a'][tau][h+1]
            s_f_h_plus_one = (self.dataset['state'][tau][h+1] == self.fail_state)
            #print(self.action_space)
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one[idx], h+1, s_f_h_plus_one)
                                for idx in range(len(self.action_space))]
            #print(len(Q_tau_h_plus_one))
            V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
            V_h_plus_one = np.hstack((V_h_plus_one, V_tau_h_plus_one))
        alpha = np.min(V_h_plus_one) + self.rho
        w_h = Lambda_h_inverse @ Phi_h.T @ np.minimum(V_h_plus_one, alpha)
        return w_h

    def update_Q(self):
        # Backward induction
        self.w = [None for _ in range(self.H)] # innitialize weights w
        for h in range(self.H-1, -1, -1):
            # calculate Lambda_h
            for n in range(self.dataset['k']):
                feature_temp = self.dataset['phi'][n][h]
                self.Lambda[h] += np.outer(feature_temp, feature_temp)
            # update w_h
            w_h = np.zeros(4)
            nu_h = np.zeros(4)
            if h == self.H - 1:
                w_h = self.theta[h]
            else:
                nu_h = self.get_nu_h(h, rho=self.Rho[h])
                w_h = self.theta[h] + nu_h
            self.w[h] = w_h
            
class SRPVI_KL():
    def __init__(self, A, beta, H, lam, dataset, Rho, theta, rho=0.2, fail_state=None):
        self.lam = lam
        self.H = H
        self.action_space = A
        self.beta = beta
        self.w = [np.zeros(4) for _ in range(self.H)]
        self.Lambda = [self.lam * np.diag(np.ones(4)) for _ in range(self.H)]
        self.Rho = Rho
        self.theta = theta
        self.fail_state = fail_state
        self.dataset = dataset
        self.rho = rho
        
    def get_action(self, phi_a, h):
        Q_h = [self.get_Q_func(phi_a[idx], h) for idx in range(len(self.action_space))]
        return self.action_space[np.argmax(Q_h)]

    def get_Q_func(self, phi, h, s_f=False):
        if s_f == True:
            return 0
        else:
            Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
            penalty = self.beta * np.sqrt(phi @ np.diag(np.diagonal(Lambda_h_inverse)) @ phi)
            Q_h = np.min([(self.w[h] @ phi - penalty), self.H - h])
            return Q_h

    def get_nu_h(self, h, rho):
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        nu_h = np.zeros(4)
        Phi_h = np.zeros((0,4))
        V_h_plus_one = np.zeros(0)
        
        for tau in range(self.dataset['k']):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h = np.vstack((Phi_h, phi_tau_h))
            phi_tau_h_plus_one = self.dataset['phi_a'][tau][h+1]
            s_f_h_plus_one = (self.dataset['state'][tau][h+1] == self.fail_state)
            #print(self.action_space)
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one[idx], h+1, s_f_h_plus_one)
                                for idx in range(len(self.action_space))]
            #print(len(Q_tau_h_plus_one))
            V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
            V_h_plus_one = np.hstack((V_h_plus_one, V_tau_h_plus_one))
        w_h = Lambda_h_inverse @ Phi_h.T @ np.exp(-V_h_plus_one / self.rho)
        return - self.rho * np.log(np.maximum(w_h, np.exp(-self.H/self.rho)))

    def update_Q(self):
        # Backward induction
        self.w = [None for _ in range(self.H)] # innitialize weights w
        for h in range(self.H-1, -1, -1):
            # calculate Lambda_h
            for n in range(self.dataset['k']):
                feature_temp = self.dataset['phi'][n][h]
                self.Lambda[h] += np.outer(feature_temp, feature_temp)
            # update w_h
            w_h = np.zeros(4)
            nu_h = np.zeros(4)
            if h == self.H - 1:
                w_h = self.theta[h]
            else:
                nu_h = self.get_nu_h(h, rho=self.Rho[h])
                w_h = self.theta[h] + nu_h
            self.w[h] = w_h
            
class SRPVI_xi2():
    def __init__(self, A, beta, H, lam, dataset, Rho, theta, rho=0.2, fail_state=None):
        self.lam = lam
        self.H = H
        self.action_space = A
        self.beta = beta
        self.w = [np.zeros(4) for _ in range(self.H)]
        self.Lambda = [self.lam * np.diag(np.ones(4)) for _ in range(self.H)]
        self.Rho = Rho
        self.theta = theta
        self.fail_state = fail_state
        self.dataset = dataset
        self.rho = rho
        
    def get_action(self, phi_a, h):
        Q_h = [self.get_Q_func(phi_a[idx], h) for idx in range(len(self.action_space))]
        return self.action_space[np.argmax(Q_h)]

    def get_Q_func(self, phi, h, s_f=False):
        if s_f == True:
            return 0
        else:
            Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
            penalty = self.beta * np.sqrt(phi @ np.diag(np.diagonal(Lambda_h_inverse)) @ phi)
            Q_h = np.min([(self.w[h] @ phi - penalty), self.H - h])
            return Q_h

    def get_nu_h(self, h, rho):
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        nu_h = np.zeros(4)
        Phi_h = np.zeros((0,4))
        V_h_plus_one = np.zeros(0)
        for tau in range(self.dataset['k']):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h = np.vstack((Phi_h, phi_tau_h))
            phi_tau_h_plus_one = self.dataset['phi_a'][tau][h+1]
            s_f_h_plus_one = (self.dataset['state'][tau][h+1] == self.fail_state)
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one[idx], h+1, s_f_h_plus_one)
                                for idx in range(len(self.action_space))]
            V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
            V_h_plus_one = np.hstack((V_h_plus_one, V_tau_h_plus_one))
        for i in range(4):
            def z_alpha_i(alpha):
                # compact formular for z
                z = Lambda_h_inverse @ Phi_h.T @ np.minimum(V_h_plus_one, alpha)
                z_2 = Lambda_h_inverse @ Phi_h.T @ (np.minimum(V_h_plus_one, alpha))**2
                return z[i] +  z[i]**2 / (4 * self.rho) - z_2[i] / (4 * self.rho)
            result = minimize(z_alpha_i, self.H/2, method='Nelder-Mead', bounds=[(0, self.H)])
            nu_h[i] = result.fun
        return nu_h

    def update_Q(self):
        # Backward induction
        self.w = [None for _ in range(self.H)] # innitialize weights w
        for h in range(self.H-1, -1, -1):
            # calculate Lambda_h
            for n in range(self.dataset['k']):
                feature_temp = self.dataset['phi'][n][h]
                self.Lambda[h] += np.outer(feature_temp, feature_temp)
            # update w_h
            w_h = np.zeros(4)
            nu_h = np.zeros(4)
            if h == self.H - 1:
                w_h = self.theta[h]
            else:
                nu_h = self.get_nu_h(h, rho=self.Rho[h])
                w_h = self.theta[h] + nu_h
            self.w[h] = w_h