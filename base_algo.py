import numpy as np
import math
from mpmath import *
from decimal import *

#design matrix
cluster_1 = np.random.normal(-5, 0.5, (4, 20))
cluster_2 = np.random.normal(3, 0.09, (3, 20))
cluster_3 = np.random.normal(1, 0.9, (6, 20))
cluster_4 = np.random.normal(-2, 0.3, (2, 20))
discriminating_data = np.concatenate((cluster_1, cluster_2, cluster_3, cluster_4))
non_discriminating_data = np.random.normal(0, 1, (15, 980))
X = np.concatenate( (discriminating_data, non_discriminating_data), axis=1)
(n,p) = X.shape

#intialization
gamma = np.zeros(p)
gamma[np.random.choice(np.arange(0, 1000), 10)] = 1
c = np.arange(1, n+1)
gamma_total_iter = 10

#HYPERPARAMETERS

#Prior of the model on the discriminatory variables : a gaussian vector
mu_0 = np.array([np.median(X[:,j]) for j in range(p)]).reshape(p, 1)  #mean for the gaussian vector
h_1 = 1000   #multiplicatory coefficient for its variance-covariance Sigma
delta = 3   #mean of the Inverse Wishart prior for Sigma
kappa_1 = 0.0007   #variance multiplicator for the variance covariance matrix of the Inverse Wishart
Q_1 = kappa_1 * np.identity(p) #variance covariance matrix of the inverse Wishart

#Prior of the model on non-discriminatory variables : a gaussian vector
h_0 = 100  #multiplicatory coefficient for its variance-covariance Omega
a=3        #first parameter of the Inverse Gamma prior on the constant variance sigma² of the non discriminatory (and assumed independent) elements
b = 0.2 #second parameter of the Inverse Gamma prior 

#Prior of gamma
omega = 10/p

def prior_gamma(gamma):
    p = 1
    for j in range(p):
        gamma_j = gamma[j]
        p *= omega**gamma_j*(1-omega)**gamma_j
    return p

def likelihood(X, gamma, c):
    # L = )  #pb : gets 0
    L = 1 #we remove constants with regard to gamma and c
    K = len(np.unique(c))
    p_gamma = int(np.sum(gamma))
    gamma_indices = np.argwhere(gamma).transpose()[0]
    gammaC_indices = np.argwhere(gamma==0).transpose()[0]
    mu_0gamma = mu_0[gamma_indices]
    mu_0gammaC = mu_0[gammaC_indices]
    Q_1gamma = Q_1[gamma_indices, :][:, gamma_indices]   #pas sûr de cette définition


    for k in range(1, K+1):
        C_k = np.argwhere(c==k)
        n_k = len(C_k)
        x_kgamma = X[k-1, gamma_indices]

        H_kgamma = (h_1 * n_k + 1)**(-p_gamma/2)
        for j in range(1, p_gamma + 1):
            H_kgamma *= math.gamma( (n_k + delta + p_gamma -j)/2) / math.gamma( (delta + p_gamma -j)/2 )

        H_0gammaC = (h_0*n + 1)**( -(p - p_gamma)/2 ) *b**( a*(p-p_gamma) )
        for j in range(1, p - p_gamma + 1):
            H_0gammaC *= math.gamma(a+n/2)/math.gamma(a)   #pb: get 0

        S_kgamma = n_k/(h_1*n_k +1)*(mu_0gamma - np.mean(x_kgamma))*np.transpose((mu_0gamma - np.mean(x_kgamma)))
        for i in C_k:
            x_igamma = X[i, gamma_indices]
            S_kgamma += (x_igamma - np.mean(x_kgamma))*np.transpose(x_igamma - np.mean(x_kgamma))

        S_0gammaC = 1
        for j in range(1, p - p_gamma + 1):
            sum_x = 0
            j_gammaC = gammaC_indices[j-1] #jth non discriminatory variable
            mu_0jgammaC = mu_0gammaC[j-1]
            for i in range(1, n+1):
                x_ijgammaC = X[i-1, j_gammaC]
                x_jgammaC = np.mean(X[:, j_gammaC])
                sum_x += (x_ijgammaC - np.mean(x_jgammaC))**2
            S_0gammaC *= b + 1/2*(sum_x + n/(h_0*n+1)*(mu_0jgammaC - np.mean(x_jgammaC))**2)  #pb : get inf

        L *= H_kgamma * np.linalg.det(Q_1gamma)**( (delta + p_gamma-1)/2 ) * np.linalg.det(Q_1gamma + S_kgamma)**( -(n_k + delta + p_gamma -1)/2 )
        print(L) #pb : gets 0
    L *= H_0gammaC * S_0gammaC**(- (a+n/2) ) 
    return L

def conditional_aposteriori_gamma(X, gamma, c):
    return likelihood(X, gamma, c) * prior_gamma(gamma)


def gamma_single_iter(gamma):
    """ Stochastic update Metropolis"""
    gamma_size = len(gamma)

    #stochastic update
    random = np.random.random()
    gamma_new = gamma.copy()
    if random < 1/2:
        #We pick an element of gamma and change its value
        index = np.random.randint(0, gamma_size)
        gamma_new[index] = abs(gamma[index] - 1)

    else: 
        #we swap a 0 and a 1
        gamma_zeros = np.argwhere(gamma==0)
        gamma_ones = np.argwhere(gamma)
        pick_zero = np.random.choice(gamma_zeros)
        pick_one = np.random.choice(gamma_ones)
        gamma_new[pick_zero] = 1 
        gamma_new[pick_one] = 0
    
    #apply metropolis probability of acceptance of the new array
    random = np.random.random()
    decision_threshold = min(1, conditional_aposteriori_gamma(X, gamma_new, c)/conditional_aposteriori_gamma(X, gamma, c))
    if random <= decision_threshold:
        gamma = gamma_new
    return gamma


for iter in gamma_total_iter:
    gamma = gamma_single_iter(gamma)


