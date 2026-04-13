# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

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

def D1(p1, p2, eps, s):
    '''Demand for firm 1: Fresh demand + return demand.
    Fresh demand: the demand from consumers who buy from firm 1 immediately upon visiting it - so
    the sum of consumers who buy immediately when seeing firm 1 first and those who search to firm 1 and then buy
    immediately when seeing firm 2 first.
    Return demand: the demand from consumers who buy from firm 1  after searching to firm 2 first and then returning 
    to firm 1.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    z_1 = z1(p1, eps, s)
    z_2 = z2(p2, s)
    D1F = 1/2 *(1 + eps - p1 - z_2) + 1/2 * ((p2+z_1)*(1 + eps - p1) - z_1**2/2)
    D1R = 1/2 * (z_2 * (2*p2 + z_2/2))
    D1 = D1F + D1R
    return D1

def D2(p1, p2, eps, s):
    '''Demand for firm 2: Fresh demand + return demand.
    Fresh demand: the demand from consumers who buy from firm 2 immediately upon visiting it - so
    the sum of consumers who buy immediately when seeing firm 2 first and those who search to firm 2 and then buy
    immediately when seeing firm 1 first.
    Return demand: the demand from consumers who buy from firm 2  after searching to firm 1 first and then returning 
    to firm 2.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    z_1 = z1(p1, eps, s)
    z_2 = z2(p2, s)
    D2F = 1/2 * (1 - z_1 - p2) + 1/2 * ((p1 - eps + z_2) * (1 - p2) - z_2**2/2)
    D2R = 1/2 * (2*z_1 * (p1 - eps) + z_1**2/2)
    D2 = D2F + D2R
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

def check_interior(p1, p2, eps, s):
    z_1 = z1(p1, eps, s)
    z_2 = z2(p2, s)
    
    cond_z = (z_1 > 0) and (z_2 > 0)
    
    cond_support1 = (eps - p1 <= 0) and (1 + eps - p1 >= 0)
    cond_support2 = (-p2 <= 0) and (1 - p2 >= 0)
    
    return {
        "z_positive": cond_z,
        "support_ok": cond_support1 and cond_support2,
        "interior": cond_z and cond_support1 and cond_support2
    }

def consumer_surplus(p1, p2, eps, s):
    EU_1 = EU1(p1, p2, eps, s)
    EU_2 = EU2(p1, p2, eps, s)
    
    return 0.5 * EU_1 + 0.5 * EU_2

def producer_surplus(p1, p2, eps, s):
    return profit1(p1, p2, eps, s) + \
           profit2(p1, p2, eps, s)

def total_welfare(p1, p2, eps, s):
    return consumer_surplus(p1, p2, eps, s) + \
           producer_surplus(p1, p2, eps, s)