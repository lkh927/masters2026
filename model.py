# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

# PARAMETERS #
# eps
# s
# theta

# CONSUMER PAYOFF #
def u1(p1, v1, eps):
    '''The utility function for consumers visiting firm 1. 
        p1: Price charged by firm 1.
        v1: Match value with firm 1, which is drawn from a uniform distribution on [0,1].
        eps: Preference shifter inducing the natural preference for firm 1.'''
    u1 = v1 + eps - p1
    return u1

def u2(p2, v2):
    '''The utility function for consumers visiting firm 2.
        p2: Price charged by firm 2.
        v2: Match value with firm 2, which is drawn from a uniform distribution on [0,1].'''
    u2 = v2 - p2
    return u2

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

def profit1(p1, p2, eps, s):
    return p1 * D1(p1, p2, eps, s)

def profit2(p1, p2, eps, s):
    return p2 * D2(p1, p2, eps, s)

def BR1(p2, eps, s):
    obj = lambda p1: -profit1(p1, p2, eps, s)
    
    res = minimize_scalar(obj, bounds=(0, 1+eps), method='bounded')
    return res.x

def BR2(p1, eps, s):
    obj = lambda p2: -profit2(p1, p2, eps, s)
    
    res = minimize_scalar(obj, bounds=(0, 1+eps), method='bounded')
    return res.x

def solve_equilibrium(eps, s, p1_init=0.5, p2_init=0.5, tol=1e-6, max_iter=500):
    
    p1, p2 = p1_init, p2_init
    
    for i in range(max_iter):
        p1_new = BR1(p2, eps, s)
        p2_new = BR2(p1_new, eps, s)
        
        if max(abs(p1_new - p1), abs(p2_new - p2)) < tol:
            return p1_new, p2_new, True
        
        p1, p2 = p1_new, p2_new
    
    return p1, p2, False

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
    
    theta = theta_star(p1, p2, eps, s)

    return {
        "p1": p1,
        "p2": p2,
        "theta": theta,
        "converged": False,
        "iterations": max_iter
    }

def check_interior(p1, p2, eps, s):
    z_1 = z1(p1, eps, s)
    z_2 = z2(p2, s)
    theta = theta_star(p1, p2, eps, s)
    
    cond_z = (z_1 >= 0) and (z_2 >= 0)
    cond_theta = (theta > 0) and (theta < 1)
    
    cond_support1 = (eps - p1 <= 0) and (1 + eps - p1 >= 0)
    cond_support2 = (-p2 <= 0) and (1 - p2 >= 0)
    
    return {
        "z_positive": cond_z,
        "theta_interior": cond_theta,
        "support_ok": cond_support1 and cond_support2,
        "interior": cond_z and cond_theta and cond_support1 and cond_support2
    }