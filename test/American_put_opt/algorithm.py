import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product
import time

class meta_algorithm():
    def __init__(self, d, beta, H, lam, dataset, Rho):
        self.lam = lam
        self.H = H
        self.d = d
        self.K = len(dataset['S'])
        self.beta = beta
        self.w = [np.zeros(self.d) for _ in range(self.H)]
        self.Lambda = [self.lam * np.diag(np.ones(self.d)) for _ in range(self.H)]
        self.Rho = Rho
        self.dataset = dataset

    def get_action(self, phi, h, current_state):
        #print(self.K - current_state)
        Q_exercising = max([0], self.K - current_state)
        #print(Q_exercising)
        Q_not_exercising = self.get_Q_func(phi, h)
        Q = [Q_not_exercising, Q_exercising[0]]
        #print(Q)
        return np.argmax(Q)

    def get_Q_func(self, phi, h):
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        penalty = self.beta * np.sqrt(phi @ np.diag(np.diagonal(Lambda_h_inverse)) @ phi)
        Q_h = np.min([(self.w[h] @ phi - penalty), self.H - h])
        return Q_h

    def get_nu_h(self, h, rho):
        pass

    def update_Q(self):
        # Backward induction
        self.w = [None for _ in range(self.H)] # innitialize weights w
        for h in range(self.H-1, -1, -1):
            # calculate Lambda_h
            #start_time = time.time()
            for n in range(self.K):
                feature_temp = self.dataset['phi'][n][h]
                self.Lambda[h] += np.outer(feature_temp, feature_temp)
            #print('Time1:', time.time() - start_time)
            # update w_h
            w_h = np.zeros(self.d)
            nu_h = np.zeros(self.d)
            if h == self.H - 1:
                w_h = np.zeros(self.d)
            else:
                w_h = self.get_nu_h(h, rho=self.Rho)
            self.w[h] = w_h
            #end_time = time.time()
            #print('Time:', end_time - start_time)

class PEVI(meta_algorithm):
    def __init__(self, d, beta, H, lam, dataset, Rho):
        self.lam = lam
        self.H = H
        self.d = d
        self.K = len(dataset['S'])
        self.beta = beta
        self.w = [np.zeros(self.d) for _ in range(self.H)]
        self.Lambda = [self.lam * np.diag(np.ones(self.d)) for _ in range(self.H)]
        self.dataset = dataset
   
    def update_Q(self):
        # Backward induction
        self.w = [None for _ in range(self.H)] #initialize weights w
        for h in range(self.H-1, -1, -1):
            # calculate Lambda_h
            for n in range(self.K):
                feature_temp = self.dataset['phi'][n][h]
                self.Lambda[h] += np.outer(feature_temp, feature_temp)
            #  calculate w_h
            Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
            w_h = np.zeros(self.d)
            if h == self.H - 1:
                w_h = np.zeros(self.d)
            else:
                for tau in range(self.K):
                    phi_tau_h = self.dataset['phi'][tau][h]
                    phi_tau_h_plus_one = self.dataset['phi'][tau][h+1]
                    s_tau_h_plus_one = self.dataset['S'][tau][h+1]
                    Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one, h+1), max(0, (self.K - s_tau_h_plus_one)[0])]
                    V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
                    w_h += Lambda_h_inverse @ (phi_tau_h * (V_tau_h_plus_one))
            self.w[h] = w_h
    

# distributional robust algorithms
class DRPVI(meta_algorithm):
    def __init__(self, d, beta, H, lam, dataset, Rho):
        super().__init__(d, beta, H, lam, dataset, Rho)
        
    def get_nu_h(self, h, rho):
        nu_h = np.zeros(self.d)
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        Phi_h = np.zeros((0, self.d))
        V_h_plus_one = np.zeros(0)
        #start = time.time()
        Phi_h = []
        for tau in range(self.K):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h.append(phi_tau_h)
            phi_tau_h_plus_one = self.dataset['phi'][tau][h+1]
            s_tau_h_plus_one = self.dataset['S'][tau][h+1]
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one, h+1), max(0, (self.K - s_tau_h_plus_one)[0])]
            V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
            V_h_plus_one = np.hstack((V_h_plus_one, V_tau_h_plus_one))
        #print(V_h_plus_one.shape)
        Phi_h = np.vstack(Phi_h)
        
        for i in range(self.d):
            def z_alpha_i(alpha):
                # compact formular for z
                z = Lambda_h_inverse @ Phi_h.T @ np.minimum(V_h_plus_one, alpha)
                return -z[i] + rho * alpha
            result = minimize(z_alpha_i, self.H/2, method='Nelder-Mead', bounds=[(0, self.H)])
            nu_h[i] = - result.fun
        return nu_h
            
class DRVI_L(meta_algorithm):
    def __init__(self, d, beta, H, lam, dataset, Rho):
        super().__init__(d, beta, H, lam, dataset, Rho)

    def get_nu_h(self, h, rho):
        nu_h = np.zeros(self.d)
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        Phi_h = np.zeros((0, self.d))
        V_h_plus_one = np.zeros(0)
        #start = time.time()
        Phi_h = []
        for tau in range(self.K):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h.append(phi_tau_h)
            phi_tau_h_plus_one = self.dataset['phi'][tau][h+1]
            s_tau_h_plus_one = self.dataset['S'][tau][h+1]
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one, h+1), max(0, (self.K - s_tau_h_plus_one)[0])]
            V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
            V_h_plus_one = np.hstack((V_h_plus_one, V_tau_h_plus_one))
        Phi_h = np.vstack(Phi_h)
        
        for i in range(self.d):
            def z_alpha_i(alpha):
                # compact formular for z
                z = Lambda_h_inverse @ Phi_h.T @ (np.exp(-V_h_plus_one / alpha) - 1)
                z_clipped = np.clip(z[i] + 1, a_min=1e-6, a_max=None)
                return alpha * np.log(z_clipped)  + alpha * rho
            result = minimize(z_alpha_i, self.H/2, method='Nelder-Mead', bounds=[(1e-6, self.H)])
            nu_h[i] = - result.fun
            #print(nu_h[i])
        return nu_h

class VA_DRPVI(meta_algorithm):
    def __init__(self, pre_agent, d, beta, H, lam, dataset, Rho):
        super().__init__(d, beta, H, lam, dataset, Rho)
        self.pre_agent = pre_agent
        self.variance = {}
        
    def get_variance_coefficient(self, h):
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        z1_h = np.zeros(self.d)
        z2_h = np.zeros(self.d)
        Phi_h = np.zeros((0,self.d))
        V_h_plus_one = np.zeros(0)
        for tau in range(self.K):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h = np.vstack((Phi_h, phi_tau_h))
            phi_tau_h_plus_one = self.dataset['phi'][tau][h+1]
            s_tau_h_plus_one = self.dataset['S'][tau][h+1]
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one, h+1), max(0, (self.K - s_tau_h_plus_one)[0])]
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

    def get_nu_h(self, h, rho, variance):
        nu_h = np.zeros(self.d)
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        Phi_h = np.zeros((0, self.d))
        V_h_plus_one = np.zeros(0)
        #start = time.time()
        Phi_h = []
        for tau in range(self.K):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h.append(phi_tau_h)
            phi_tau_h_plus_one = self.dataset['phi'][tau][h+1]
            s_tau_h_plus_one = self.dataset['S'][tau][h+1]
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one, h+1), max(0, (self.K - s_tau_h_plus_one)[0])]
            V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
            V_h_plus_one = np.hstack((V_h_plus_one, V_tau_h_plus_one))
        Phi_h = np.vstack(Phi_h)
        for i in range(self.d):
            def z_alpha_i(alpha):
                # compact formular for z
                z = Lambda_h_inverse @ Phi_h.T @ (np.minimum(V_h_plus_one, alpha) / variance)
                return -z[i] + rho * alpha
            result = minimize(z_alpha_i, self.H/2, method='Nelder-Mead', bounds=[(0, self.H)])
            nu_h[i] = - result.fun
        return nu_h
        
    def update_Q(self):
        # Backward induction
        self.w = [None for _ in range(self.H)] # initialize weights w
        for h in range(self.H-1, -1, -1):
            # calculate Lambda_h
            if h == self.H-1:
                for n in range(self.K):
                    feature_temp = self.dataset['phi'][n][h]
                    self.Lambda[h] += np.outer(feature_temp, feature_temp)
            else:
                z1_h, z2_h = self.get_variance_coefficient(h)
                self.variance[str(h)] = np.zeros(self.K)
                for n in range(self.K):
                    feature_temp = self.dataset['phi'][n][h]
                    variance_temp = self.estimated_variance(feature_temp, z1_h, z2_h)
                    self.variance[str(h)][n] = variance_temp
                    self.Lambda[h] += np.outer(feature_temp, feature_temp) / variance_temp
            # update w_h
            w_h = np.zeros(self.d)
            nu_h = np.zeros(self.d)
            if h == self.H - 1:
                w_h = np.zeros(self.d)
            else:
                variance = self.variance[str(h)]
                nu_h = self.get_nu_h(h, rho = self.Rho, variance = variance)
                w_h = nu_h
            self.w[h] = w_h

# regularized distributional robust algorithms
                                    
class R2PVI_TV(meta_algorithm):
    def __init__(self, d, beta, H, lam, dataset, Rho):
        super().__init__(d, beta, H, lam, dataset, Rho)
        
    def get_nu_h(self, h, rho):
        
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        Phi_h = np.zeros((0, self.d))
        V_h_plus_one = []
        #start = time.time()
        Phi_h = []
        for tau in range(self.K):
            #print(1)
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h.append(phi_tau_h)
            phi_tau_h_plus_one = self.dataset['phi'][tau][h+1]
            s_tau_h_plus_one = self.dataset['S'][tau][h+1]
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one, h+1), max(0, (self.K - s_tau_h_plus_one)[0])]
            V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
            V_h_plus_one.append(V_tau_h_plus_one)
        V_h_plus_one = np.hstack(V_h_plus_one)
        #print(V_h_plus_one.shape)
        Phi_h = np.vstack(Phi_h)
        #print(f"w_h computation time: {time.time() - start:.5f} seconds")
        alpha = np.min(V_h_plus_one) + rho

        w_h = Lambda_h_inverse @ Phi_h.T @ np.minimum(V_h_plus_one, alpha)

        
        return w_h
    
class R2PVI_KL(meta_algorithm):
    def __init__(self, d, beta, H, lam, dataset, Rho):
        super().__init__(d, beta, H, lam, dataset, Rho)

    def get_nu_h(self, h, rho):
        nu_h = np.zeros(self.d)
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        Phi_h = np.zeros((0, self.d))
        V_h_plus_one = []
        #start = time.time()
        Phi_h = []
        for tau in range(self.K):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h.append(phi_tau_h)
            phi_tau_h_plus_one = self.dataset['phi'][tau][h+1]
            s_tau_h_plus_one = self.dataset['S'][tau][h+1]
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one, h+1), max(0, (self.K - s_tau_h_plus_one)[0])]
            V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
            V_h_plus_one.append(V_tau_h_plus_one)
        Phi_h = np.vstack(Phi_h)
        V_h_plus_one = np.hstack(V_h_plus_one)
        w_h = Lambda_h_inverse @ Phi_h.T @ np.exp(-V_h_plus_one / rho)
        #print(w_h)
        return - rho * np.log(np.maximum(w_h, np.exp(-self.H/rho)))
    
class R2PVI_xi2(meta_algorithm):
    def __init__(self, d, beta, H, lam, dataset, Rho):
        super().__init__(d, beta, H, lam, dataset, Rho)
        
    def get_nu_h(self, h, rho):
        nu_h = np.zeros(self.d)
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        Phi_h = np.zeros((0, self.d))
        V_h_plus_one = []
        #start = time.time()
        Phi_h = []
        for tau in range(self.K):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h.append(phi_tau_h)
            phi_tau_h_plus_one = self.dataset['phi'][tau][h+1]
            s_tau_h_plus_one = self.dataset['S'][tau][h+1]
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one, h+1), max(0, (self.K - s_tau_h_plus_one)[0])]
            V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
            V_h_plus_one.append(V_tau_h_plus_one)
        Phi_h = np.vstack(Phi_h)
        V_h_plus_one = np.hstack(V_h_plus_one)
        for i in range(self.d):
            def z_alpha_i(alpha):
                # compact formular for z
                z = Lambda_h_inverse @ Phi_h.T @ np.minimum(V_h_plus_one, alpha)
                z_2 = Lambda_h_inverse @ Phi_h.T @ (np.minimum(V_h_plus_one, alpha))**2
                return -(z[i] +  z[i]**2 / (4 * rho) - z_2[i] / (4 * rho))
            result = minimize(z_alpha_i, self.H/2, method='Nelder-Mead', bounds=[(0, self.H)])
            nu_h[i] = -result.fun
        return nu_h
    
class VA_R2PVI_TV(meta_algorithm):
    def __init__(self, pre_agent, d, beta, H, lam, dataset, Rho):
        super().__init__(d, beta, H, lam, dataset, Rho)
        self.pre_agent = pre_agent
        self.variance = {}
        
    def get_variance_coefficient(self, h):
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        z1_h = np.zeros(self.d)
        z2_h = np.zeros(self.d)
        Phi_h = np.zeros((0,self.d))
        V_h_plus_one = np.zeros(0)
        for tau in range(self.K):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h = np.vstack((Phi_h, phi_tau_h))
            phi_tau_h_plus_one = self.dataset['phi'][tau][h+1]
            s_tau_h_plus_one = self.dataset['S'][tau][h+1]
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one, h+1), max(0, (self.K - s_tau_h_plus_one)[0])]
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

    def get_nu_h(self, h, rho, variance):
        nu_h = np.zeros(self.d)
        Lambda_h_inverse = np.linalg.inv(self.Lambda[h])
        Phi_h = np.zeros((0, self.d))
        V_h_plus_one = np.zeros(0)
        #start = time.time()
        Phi_h = []
        for tau in range(self.K):
            phi_tau_h = self.dataset['phi'][tau][h]
            Phi_h.append(phi_tau_h)
            phi_tau_h_plus_one = self.dataset['phi'][tau][h+1]
            s_tau_h_plus_one = self.dataset['S'][tau][h+1]
            Q_tau_h_plus_one = [self.get_Q_func(phi_tau_h_plus_one, h+1), max(0, (self.K - s_tau_h_plus_one)[0])]
            V_tau_h_plus_one = np.max(Q_tau_h_plus_one)
            V_h_plus_one = np.hstack((V_h_plus_one, V_tau_h_plus_one))
        Phi_h = np.vstack(Phi_h)
        alpha = np.min(V_h_plus_one) + rho
        w_h = Lambda_h_inverse @ Phi_h.T @ (np.minimum(V_h_plus_one, alpha) / variance)
        return w_h
        
    def update_Q(self):
        # Backward induction
        self.w = [None for _ in range(self.H)] # initialize weights w
        for h in range(self.H-1, -1, -1):
            # calculate Lambda_h
            if h == self.H-1:
                for n in range(self.K):
                    feature_temp = self.dataset['phi'][n][h]
                    self.Lambda[h] += np.outer(feature_temp, feature_temp)
            else:
                z1_h, z2_h = self.get_variance_coefficient(h)
                self.variance[str(h)] = np.zeros(self.K)
                for n in range(self.K):
                    feature_temp = self.dataset['phi'][n][h]
                    variance_temp = self.estimated_variance(feature_temp, z1_h, z2_h)
                    self.variance[str(h)][n] = variance_temp
                    self.Lambda[h] += np.outer(feature_temp, feature_temp) / variance_temp
            # update w_h
            w_h = np.zeros(self.d)
            nu_h = np.zeros(self.d)
            if h == self.H - 1:
                w_h = np.zeros(self.d)
            else:
                variance = self.variance[str(h)]
                nu_h = self.get_nu_h(h, rho = self.Rho, variance = variance)
                w_h = nu_h
            self.w[h] = w_h