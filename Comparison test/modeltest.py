import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

# RESERVATION VALUES #
def z1(p1, eps, s):
    '''Reservation value for consumeres who visit firm 2 first. Defines the threshold match value for
    which consumers are indifferent between stopping search at firm 2 and paying s to search on to firm 1
        p1: Price charged by firm 1.
        eps: Preference shifter inducing the natural preference for firm 1.
        s: Search cost.'''
    z1 = 1 + eps - p1 - np.sqrt(2*s)
    return max(z1,0)

def z2(p2, s):
    '''Reservation value for consumeres who visit firm 1 first. Defines the threshold match value for
    which consumers are indifferent between stopping search at firm 1 and paying s to search on to firm 2
        p2: Price charged by firm 2.
        s: Search cost.'''
    z2 = 1 - p2 - np.sqrt(2*s)
    return max(z2,0)

def z1_var(p1, s, a1, b1):
    L1 = b1 - a1
    z1 = b1 - p1 - np.sqrt(2*s*L1)
    return max(z1, 0)

def z2_var(p2, s, a2, b2):
    L2 = b2 - a2
    z2 = b2 - p2 - np.sqrt(2*s*L2)
    return max(z2, 0)


# EXPECTED UTILITIES AND DISCLOSURE CUTOFF #
def EU1(p1, p2, eps, s):
    '''Expected utility of visiting firm 1 first, given that the first visit is free.
    EU1 = E[max{u1, z2, 0}]. Visiting firm 1 first happens with probability 1 if consumers decide to disclose
    and with probability 1/2 if they do not.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    z_2 = z2(p2, s)
    EU1 = z_2*(z_2 - eps + p1) + ((1 + eps - p1)**2 - z_2**2)/2
    return EU1

def EU2(p1, p2, eps, s):
    '''Expected utility of visiting firm 2 first, given that the first visit is free.
    EU2 = E[max{u2, z1, 0}]. Visiting firm 2 first happens with probability 0 if consumers decide to disclose
    and with probaiblity 0 if they do not.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    z_1 = z1(p1, eps, s)
    EU2 = z_1*(z_1 + p2) + ((1-p2)**2 - z_1**2)/2
    return EU2

def EU1_var(p1, p2, s, a1, b1, a2, b2):
    z_2 = z2_var(p2, s, a2, b2)
    L1 = b1 - a1
    return ((b1 - p1)**2 + z_2**2) / (2*L1)

def EU2_var(p1, p2, s, a1, b1, a2, b2):
    z_1 = z1_var(p1, s, a1, b1)
    L2 = b2 - a2
    return ((b2 - p2)**2 + z_1**2) / (2*L2)

def theta_star(p1, p2, eps, s):
    '''The disclosure cutoff for consumers who choose to disclose their preference for firm 1. 
    Disclosing this information comes at cost theta_i, which is the consumers' individual type, 
    capturing their privacy preferences. Consumers with theta_i ≤ theta_star disclose, while consumers
    with theta_i > theta_star do not.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''

    theta = (EU1(p1, p2, eps, s) - EU2(p1, p2, eps, s))/2
    return np.clip(theta, 0, 1)

def theta_star_var(p1, p2, s, a1, b1, a2, b2):
    theta = 0.5 * (EU1_var(p1, p2, s, a1, b1, a2, b2)
                   - EU2_var(p1, p2, s, a1, b1, a2, b2))
    return np.clip(theta, 0, 1)


# DEMAND #
# interior case relies on positive reservation values and 0 is in [eps-p_1, 1+eps-p_1] and [-p_2, 1-p_2]
def D1F(p1, p2, eps, s):
    '''Fresh demand for firm 1: the demand from consumers who buy from firm 1 immediately upon visiting it - so
    the sum of consumers who buy immediately when seeing firm 1 first and those who search to firm 1 and then buy
    immediately when seeing firm 2 first.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    thetastar = theta_star(p1, p2, eps, s)
    z_1 = z1(p1, eps, s)
    z_2 = z2(p2, s)
    D1F = (1+thetastar)/2 *(1 + eps - p1 - z_2) + (1-thetastar)/2 * ((p2+z_1)*(1 + eps - p1) - z_1**2/2)
    return D1F

def D1R(p1, p2, eps, s):
    '''Return demand for firm 1: the demand from consumers who buy from firm 1 after searching to firm 2 first
    and then returning to firm 1.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    thetastar = theta_star(p1, p2, eps, s)
    z_2 = z2(p2, s)
    D1R = (1+thetastar)/2 * (z_2 * (2*p2 + z_2/2))
    return D1R

def D1(p1, p2, eps, s):
    '''Total demand for firm 1: the sum of fresh demand and return demand.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    D1 = D1F(p1, p2, eps, s) + D1R(p1, p2, eps, s)
    return D1

def D2F(p1, p2, eps, s):
    '''Fresh demand for firm 2: the demand from consumers who buy from firm 2 immediately upon visiting it - so
    the sum of consumers who buy immediately when seeing firm 2 first and those who search to firm 2 and then buy
    immediately when seeing firm 1 first.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    thetastar = theta_star(p1, p2, eps, s)
    z_1 = z1(p1, eps, s)
    z_2 = z2(p2, s)
    D2F = (1-thetastar)/2 * (1 - z_1 - p2) + (1+thetastar)/2 * ((p1 - eps + z_2) * (1 - p2) - z_2**2/2)
    return D2F

def D2R(p1, p2, eps, s):
    '''Return demand for firm 2: the demand from consumers who buy from firm 2 after searching to firm 1 first
    and then returning to firm 2.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    thetastar = theta_star(p1, p2, eps, s)
    z_1 = z1(p1, eps, s)
    D2R = (1-thetastar)/2 * (2*z_1 * (p1 - eps) + z_1**2/2)
    return D2R

def D2(p1, p2, eps, s):
    '''Total demand for firm 2: the sum of fresh demand and return demand.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    D2 = D2F(p1, p2, eps, s) + D2R(p1, p2, eps, s)
    return D2

# HELPER CDF FUNCTIONS #
def F1(u, p1, a1, b1):
    return np.clip((u - (a1 - p1)) / (b1 - a1), 0, 1)

def F2(u, p2, a2, b2):
    return np.clip((u - (a2 - p2)) / (b2 - a2), 0, 1)

def P_u1_ge_u2(p1, p2, a1, b1, a2, b2):
    L1 = b1 - a1
    L2 = b2 - a2
    
    low1 = a1 - p1
    high1 = b1 - p1
    low2 = a2 - p2
    
    term1 = 0.5 * (high1**2 - low1**2)
    term2 = (low2) * (b1 - a1)
    
    return (term1 - term2) / (L1 * L2)

def D1_var(p1, p2, s, a1, b1, a2, b2):
    theta = theta_star_var(p1, p2, s, a1, b1, a2, b2)
    z1 = z1_var(p1, s, a1, b1)
    z2 = z2_var(p2, s, a2, b2)

    # Probabilities
    P_u1_ge_z2 = 1 - F1(z2, p1, a1, b1)
    P_u2_le_0 = F2(0, p2, a2, b2)

    # Fresh demand
    D1F = (1+theta)/2 * P_u1_ge_z2 \
        + (1-theta)/2 * P_u1_ge_u2(p1, p2, a1, b1, a2, b2)

    # Return demand (approximate but consistent)
    D1R = (1+theta)/2 * P_u2_le_0 * F1(z2, p1, a1, b1)

    return D1F + D1R

def D2F_var(p1, p2, s, a1, b1, a2, b2):
    theta = theta_star_var(p1, p2, s, a1, b1, a2, b2)
    z1 = z1_var(p1, s, a1, b1)
    z2 = z2_var(p2, s, a2, b2)

    L1 = b1 - a1
    L2 = b2 - a2

    # CDFs
    F1_0 = np.clip((0 - (a1 - p1)) / L1, 0, 1)
    F2_0 = np.clip((0 - (a2 - p2)) / L2, 0, 1)
    F2_z1 = np.clip((z1 - (a2 - p2)) / L2, 0, 1)

    # First term
    term1 = (1 - theta)/2 * (1 - F2_z1)

    # Second term (two parts)
    part1 = F1_0 * (1 - F2_0)

    part2 = ((b2 - p2)*z2 - 0.5*z2**2) / (L1 * L2)

    term2 = (1 + theta)/2 * (part1 + part2)

    return term1 + term2

def D2_var(p1, p2, s, a1, b1, a2, b2):
    theta = theta_star_var(p1, p2, s, a1, b1, a2, b2)
    z1 = z1_var(p1, s, a1, b1)

    P_u1_le_0 = F1(0, p1, a1, b1)

    D2F = D2F_var(p1, p2, s, a1, b1, a2, b2)

    D2R = (1-theta)/2 * P_u1_le_0 * F2(z1, p2, a2, b2)

    return D2F + D2R

# PROFITS #
def profit1(p1, p2, eps, s):
    return p1 * D1(p1, p2, eps, s)

def profit2(p1, p2, eps, s):
    return p2 * D2(p1, p2, eps, s)

def profit1_var(p1, p2, s, a1, b1, a2, b2):
    return p1 * D1_var(p1, p2, s, a1, b1, a2, b2)

def profit2_var(p1, p2, s, a1, b1, a2, b2):
    return p2 * D2_var(p1, p2, s, a1, b1, a2, b2)

# BEST RESPONSES #

def BR1(p2, eps, s):
    obj = lambda p1: -profit1(p1, p2, eps, s)
    
    res = minimize_scalar(obj, bounds=(0, 1+eps), method='bounded')
    return res.x

def BR2(p1, eps, s):
    obj = lambda p2: -profit2(p1, p2, eps, s)
    
    res = minimize_scalar(obj, bounds=(0, 1+eps), method='bounded')
    return res.x

def BR1_var(p2, s, a1, b1, a2, b2):
    obj = lambda p1: -profit1_var(p1, p2, s, a1, b1, a2, b2)

    res = minimize_scalar(obj, bounds=(0, b1), method='bounded')
    return res.x

def BR2_var(p1, s, a1, b1, a2, b2):
    obj = lambda p2: -profit2_var(p1, p2, s, a1, b1, a2, b2)

    res = minimize_scalar(obj, bounds=(0, b2), method='bounded')
    return res.x

# EQUILIBRIUM #
def solve_equilibrium_o(eps, s, tol=1e-6, max_iter=500):
    
    # initial guess
    p1, p2 = 0.5, 0.5
    
    for i in range(max_iter):
        p1_new = BR1(p2, eps, s)
        p2_new = BR2(p1_new, eps, s)
        
        if max(abs(p1_new - p1), abs(p2_new - p2)) < tol:
            return {
                "p1": p1_new,
                "p2": p2_new,
                "converged": True,
                "iterations": i
            }
        
        p1, p2 = p1_new, p2_new
    
    return {
        "p1": p1,
        "p2": p2,
        "converged": False,
        "iterations": max_iter
    }

def solve_equilibrium_var(s, a1, b1, a2, b2, tol=1e-6, max_iter=500):
    p1, p2 = 0.5, 0.5

    for i in range(max_iter):
        p1_new = BR1_var(p2, s, a1, b1, a2, b2)
        p2_new = BR2_var(p1_new, s, a1, b1, a2, b2)

        if max(abs(p1_new - p1), abs(p2_new - p2)) < tol:
            return {
                "p1": p1_new, 
                "p2": p2_new, 
                "converged": True
            }

        p1, p2 = p1_new, p2_new

    return {
        "p1": p1, 
        "p2": p2, 
        "converged": False
    }