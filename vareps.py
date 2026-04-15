# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar


# RESERVATION VALUES #

# TYPE A (prefers firm 1) — SAME AS BASELINE
def z1_A(p1, eps, s):
    return max(1 + eps - p1 - np.sqrt(2*s), 0)

def z2_A(p2, s):
    return max(1 - p2 - np.sqrt(2*s), 0)

# TYPE B (prefers firm 2)
def z1_B(p1, s):
    return max(1 - p1 - np.sqrt(2*s), 0)

def z2_B(p2, eps, s):
    return max(1 + eps - p2 - np.sqrt(2*s), 0)

# EXPECTED UTILITIES AND DISCLOSURE CUTOFF #

# TYPE A
def EU1_A(p1, p2, eps, s):
    z_2 = z2_A(p2, s)
    return z_2*(z_2 - eps + p1) + ((1 + eps - p1)**2 - z_2**2)/2

def EU2_A(p1, p2, eps, s):
    z_1 = z1_A(p1, eps, s)
    return z_1*(z_1 + p2) + ((1 - p2)**2 - z_1**2)/2

def theta_A_star(p1, p2, eps, s):
    return np.clip((EU1_A(p1, p2, eps, s) - EU2_A(p1, p2, eps, s))/2, 0, 1)

# TYPE B
def EU1_B(p1, p2, eps, s):
    z_2 = z2_B(p2, eps, s)
    return z_2*(z_2 + p1) + ((1 - p1)**2 - z_2**2)/2

def EU2_B(p1, p2, eps, s):
    z_1 = z1_B(p1, s)
    return z_1*(z_1 - eps + p2) + ((1 + eps - p2)**2 - z_1**2)/2

def theta_B_star(p1, p2, eps, s):
    return np.clip((EU2_B(p1, p2, eps, s) - EU1_B(p1, p2, eps, s))/2, 0, 1)

def Theta_star(p1, p2, eps, s, mu):
    return mu * theta_A_star(p1, p2, eps, s) + (1-mu) * theta_B_star(p1, p2, eps, s)

# RANKING PROBABILITIES #
def ranking_probs(p1, p2, eps, s, mu):
    thetaA = theta_A_star(p1, p2, eps, s)
    thetaB = theta_B_star(p1, p2, eps, s)
    
    non_disclosure = mu*(1-thetaA) + (1-mu)*(1-thetaB)
    
    pi1_first = mu*thetaA + 0.5 * non_disclosure
    pi2_first = (1-mu)*thetaB + 0.5 * non_disclosure
    
    return pi1_first, pi2_first

# DEMAND #
# interior case relies on positive reservation values and 0 is in [eps-p_1, 1+eps-p_1] and [-p_2, 1-p_2]
# TYPE A
def D1_A(p1, p2, eps, s):
    thetaA = theta_A_star(p1, p2, eps, s)
    z_1 = z1_A(p1, eps, s)
    z_2 = z2_A(p2, s)
    
    D1F = (1+thetaA)/2 *(1 + eps - p1 - z_2) \
        + (1-thetaA)/2 * ((p2+z_1)*(1 + eps - p1) - z_1**2/2)
    
    D1R = (1+thetaA)/2 * (z_2 * (2*p2 + z_2/2))
    
    return D1F + D1R

def D2_A(p1, p2, eps, s):
    thetaA = theta_A_star(p1, p2, eps, s)
    z_1 = z1_A(p1, eps, s)
    z_2 = z2_A(p2, s)
    
    D2F = (1-thetaA)/2 * (1 - z_1 - p2) \
        + (1+thetaA)/2 * ((p1 - eps + z_2)*(1 - p2) - z_2**2/2)
    
    D2R = (1-thetaA)/2 * (2*z_1*(p1 - eps) + z_1**2/2)
    
    return D2F + D2R

# TYPE B
def D1_B(p1, p2, eps, s):
    thetaB = theta_B_star(p1, p2, eps, s)
    z_1 = z1_B(p1, s)
    z_2 = z2_B(p2, eps, s)
    
    # firm 1 is now the "weaker" firm
    
    D1F = (1-thetaB)/2 * (1 - z_2 - p1) \
        + (1+thetaB)/2 * ((p2 - eps + z_1)*(1 - p1) - z_1**2/2)
    
    D1R = (1-thetaB)/2 * (2*z_2*(p2 - eps) + z_2**2/2)
    
    return D1F + D1R

def D2_B(p1, p2, eps, s):
    thetaB = theta_B_star(p1, p2, eps, s)
    z_1 = z1_B(p1, s)
    z_2 = z2_B(p2, eps, s)
    
    D2F = (1+thetaB)/2 * (1 + eps - p2 - z_1) \
        + (1-thetaB)/2 * ((p1+z_2)*(1 + eps - p2) - z_2**2/2)
    
    D2R = (1+thetaB)/2 * (z_1 * (2*p1 + z_1/2))
    
    return D2F + D2R

def D1_total(p1, p2, eps, s, mu):
    return mu * D1_A(p1, p2, eps, s) + (1-mu) * D1_B(p1, p2, eps, s)

def D2_total(p1, p2, eps, s, mu):
    return mu * D2_A(p1, p2, eps, s) + (1-mu) * D2_B(p1, p2, eps, s)

# FIRM PROFITS #
def profit1(p1, p2, eps, s, lam):
    return p1 * D1_total(p1, p2, eps, s, lam)

def profit2(p1, p2, eps, s, lam):
    return p2 * D2_total(p1, p2, eps, s, lam)

# BEST RESPONSE FUNCTIONS #
def BR1(p2, eps, s, lam):
    obj = lambda p1: -profit1(p1, p2, eps, s, lam)
    res = minimize_scalar(obj, bounds=(0, 1+eps), method='bounded')
    return res.x

def BR2(p1, eps, s, lam):
    obj = lambda p2: -profit2(p1, p2, eps, s, lam)
    res = minimize_scalar(obj, bounds=(0, 1+eps), method='bounded')
    return res.x


# EQUILIBRIUM SOLVER # 
def solve_equilibrium(eps, s, mu, p1_init=0.5, p2_init=0.5, tol=1e-6, max_iter=500):
    
    p1, p2 = p1_init, p2_init
    
    for i in range(max_iter):
        p1_new = BR1(p2, eps, s, mu)
        p2_new = BR2(p1_new, eps, s, mu)
        
        if max(abs(p1_new - p1), abs(p2_new - p2)) < tol:
            return p1_new, p2_new, True
        
        p1, p2 = p1_new, p2_new
    
    return p1, p2, False

