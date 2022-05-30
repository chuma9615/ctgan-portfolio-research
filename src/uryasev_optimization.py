import numpy as np
from scipy.optimize import linprog
import pandas as pd



class UryasevOptimization():
    """
    Represents an Uryasev & Rockafeller optimization.
    """
    def __init__(self, alpha, cvar, bounds):
        self.alpha = alpha
        self.cvar = cvar
        self.bounds = bounds
    
    def get_optimal_portfolio(self, sample, density=None):
        '''
        Generates and resolves Uryasev's optimization problem.
        '''
        # define the probabilities for each window, all equal in this simple model
        if density is None:
                density = np.ones(len(sample))/len(sample)
        # our expected return will be the mean of the distribution
        mu = sample.mean(axis=0)
        # start building the matrix for the linear optimization
        J = sample.shape[0]
        n = sample.shape[1]
        A = np.zeros((1 + J + J + 1 , 1 + n + J))
        b = np.zeros((1 + J + J + 1))
        i = 0
        c = np.zeros(1 + n + J)
        v = [(0, None) for x in range((1 + n + J))]
        
        # build bounds
        for i in range(n):
                v[1 + i] = self.bounds
        # build objetive function
        for i in range(n):
                c[1 + i] = - mu[i]   
        # build restrictions
        for i in range(np.size(A, axis=0)):
                for k in range(np.size(A, axis=1)):
                        # cvar restriction
                        if i==0:
                                b[i] = self.cvar
                                if k==0:
                                        A[i, k] = 1
                                elif k in range(1 + n, 1 + n + J):
                                        A[i, k] = (1 - self.alpha)**-1*density[k -(1 + n)]
                        # select samples under threshold
                        elif i in range(1, 1+J):
                                b[i] = 0
                                if k == 0:
                                        A[i,k] = -1
                                elif k in range(1, 1 + n):
                                        A[i,k] = -sample[i - 1, k - 1]
                                elif k in range(1 + n, 1 + n + J) and (i - 1) == (k - (1 + n)):
                                        A[i,k] = -1
                        # z non-negativity
                        elif i in range(1 + J, 1 + J + J):
                                b[i] = 0
                                if k in range(1 + n, 1 + n + J) and (i-(1 + J + 1 + 2*n))==(k - (1 + n)):
                                        A[i,k] = -1
                        # 100% max investment (non-leveraged fund)
                        elif i in range(1 + J + J, 1 + J + J + 1):
                                b[i] = 1
                                if k in range(1, 1 + n):
                                        A[i,k] = 1
        # solve the problem
        optimal_result = linprog(c, A_ub=A, b_ub=b, bounds=v, options={"disp": False})
        optimal_portfolio = optimal_result.x[1:n+1]
        optimal_portfolio = pd.Series(optimal_portfolio)
        # remove scraps
        optimal_portfolio[optimal_portfolio<0.01] = 0
        optimal_portfolio /= optimal_portfolio.sum()
        optimal_portfolio *= 100
        return optimal_portfolio

    
