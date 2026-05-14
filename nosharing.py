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
    z1_b = 1 + eps - p1 - np.sqrt(2*s)
    z1_w = 1 - p1 - np.sqrt(2*s)
    return z1_b, z1_w

def z2(p2, eps, s):
    '''Reservation value for consumeres who visit firm 1 first. Defines the threshold match value for
    which consumers are indifferent between stopping search at firm 1 and paying s to search on to firm 2
        p2: Price charged by firm 2.
        eps: Preference shifter inducing the natural preference for firm 2.
        s: Search cost.'''
    z2_w = 1 - p2 - np.sqrt(2*s)
    z2_b = 1 + eps - p2 - np.sqrt(2*s)
    return z2_w, z2_b

    
# EXPECTED UTILITIES AND DISCLOSURE CUTOFF #
def EU1_B(p1, p2, eps, s):
    '''Expected utility of visiting firm 1 first, given that the first visit is free.
    EU1 = E[max{u1, z2, 0}]. Here for type A, who prefers firm 1.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    z_2, _ = z2(p2, eps, s)
    if z_2 > 0:
        EU1 = z_2*(z_2 - eps + p1) + ((1 + eps - p1)**2 - z_2**2)/2
    else:
         EU1 = (1 + eps - p1)**2/2
    return EU1

def EU1_W(p1, p2, eps, s):
    '''Expected utility of visiting firm 1 first, given that the first visit is free.
    EU1 = E[max{u1, z2, 0}]. Here for type B, who prefers firm 2.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    _, z_2 = z2(p2, eps, s)
    if z_2 > 0:
        EU1 = z_2*(z_2 + p1) + ((1 - p1)**2 - z_2**2)/2
    else:
         EU1 = (1 - p1)**2/2
    return EU1

def EU2_W(p1, p2, eps, s):
    '''Expected utility of visiting firm 2 first, given that the first visit is free.
    EU2 = E[max{u2, z1, 0}]. Visiting firm 2 first happens with probability 0 if consumers decide to disclose
    and with probaiblity 0 if they do not. Here for type A.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    z_1, _ = z1(p1, eps, s)
    if z_1 > 0:
        EU2 = z_1*(z_1 + p2) + ((1-p2)**2 - z_1**2)/2
    else:
        EU2 = (1 - p2)**2/2
    return EU2

def EU2_B(p1, p2, eps, s):
    '''Expected utility of visiting firm 2 first, given that the first visit is free.
    EU2 = E[max{u2, z1, 0}]. Visiting firm 2 first happens with probability 0 if consumers decide to disclose
    and with probaiblity 0 if they do not. Here for type B.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    _, z_1 = z1(p1, eps, s)
    if z_1 > 0:
        EU2 = z_1*(z_1 - eps + p2) + ((1 + eps - p2)**2 - z_1**2)/2
    else:
        EU2 = (1 + eps - p2)**2/2
    return EU2


def D1_F(p1, p2, eps, s, mu):
    z1_b, z1_w = z1(p1, eps, s)
    z2_w, z2_b = z2(p2, eps, s)

    # Firm 1 is preferred: D1_F = DBF_1 (preferred firm, F=1)
    if z1_b > 0 and z2_w > 0: # reg A (active search)
        DBF_1 = 1/2 *(1 + eps - p1 - z2_w) \
        + 1/2 * ((p2+z1_b)*(1 + eps - p1) - z1_b**2/2)
    elif z1_b > 0 and z2_w <= 0: # reg AN (no search after good match)
        DBF_1 = 1/2 *(1 + eps - p1) \
        + 1/2 * ((p2+z1_b)*(1 + eps - p1) - z1_b**2/2)
    elif z1_b <= 0 and z2_w > 0: # reg NA (no search after bad match)
        DBF_1 = 1/2 *(1 + eps - p1 - z2_w) 
    elif z1_b <=0 and z2_w <= 0: # reg N (no search)
        DBF_1 = 1/2 *(1 + eps - p1)

    # Firm 2 is preferred: D1_F = DWF_1 (non-preferred firm, F=1)
    if z1_w > 0 and z2_b > 0: # reg A (active search)
        DWF_1 = 1/2 * (1 - z2_b - p1) \
        + 1/2 * ((p2 - eps + z1_w)*(1 - p1) - z1_w**2/2)
    elif z1_w <=0 and z2_b > 0: # reg AN (no search after good match)
        DWF_1 = 1/2 * (1 - z2_b - p1)
    elif z1_w > 0 and z2_b <= 0: # reg NA (no search after bad match)
        DWF_1 = 1/2 * (1 - p1) \
        + 1/2 * ((p2 - eps + z1_w)*(1 - p1) - z1_w**2/2)
    elif z1_w <= 0 and z2_b <= 0: # reg N (no search)
        DWF_1 = 1/2 * (1 - p1)

    D1_F = mu * DBF_1 + (1-mu) * DWF_1
    return D1_F


def D1_R(p2, eps, s, mu):
    z2_w, z2_b = z2(p2, eps, s)

    # Firm 1 is preferred:
    if z2_w > 0: # reg A or AN (active search after good match)
        DBR_1 = 1/2 * (2*z2_w*p2 + z2_w**2/2)
    else: # reg NA or N (no search after good match)
        DBR_1 = 0
    
    # Firm 2 is preferred:
    if z2_b > 0: # reg A or AN (active search after bad match)
        DWR_1 = 1/2 * (2*z2_b*(p2 - eps) + z2_b**2/2)
    else: # reg NA or N (no search after bad match)
        DWR_1 = 0
    
    D1_R = mu * DBR_1 + (1-mu) * DWR_1
    return D1_R

def D1(p1, p2, eps, s, mu):
    D1F = D1_F(p1, p2, eps, s, mu)
    D1R = D1_R(p2, eps, s, mu)
    return D1F + D1R


def D2_F(p1, p2, eps, s, mu):
    z1_b, z1_w = z1(p1, eps, s)
    z2_w, z2_b = z2(p2, eps, s)

    # Firm 1 is preferred: D2_F = DWF_2 (non-preferred firm, F=2)
    if z1_b > 0 and z2_w > 0: # reg A (active search)
        DWF_2 = 1/2 *(1 - z1_b - p2) \
        + 1/2 * ((p1 - eps + z2_w)*(1 - p2) - z2_w**2/2)
    elif z1_b > 0 and z2_w <= 0: # reg AN (no search after good match)
        DWF_2 = 1/2 *(1 - z1_b - p2)
    elif z1_b <= 0 and z2_w > 0: # reg NA (no search after bad match)
        DWF_2 = 1/2 *(1 - p2) \
        + 1/2 * ((p1 - eps + z2_w)*(1 - p2) - z2_w**2/2)
    elif z1_b <=0 and z2_w <= 0: # reg N (no search)
        DWF_2 = 1/2 * (1 - p2)
    
    # Firm 2 is prererred: D2_F = DBF_2 (preferred firm, F=2)
    if z1_w > 0 and z2_b > 0: # reg A (active search)
        DBF_2 = 1/2 * (1 + eps - p2 - z1_w) \
        + 1/2 * ((p1+z2_b)*(1 + eps - p2) - z2_b**2/2)
    elif z1_w <=0 and z2_b > 0: # reg AN (no search after good match)
        DBF_2 = 1/2 * (1 + eps - p2) \
        + 1/2 * ((p1+z2_b)*(1 + eps - p2) - z2_b**2/2)
    elif z1_w > 0 and z2_b <= 0: # reg NA (no search after bad match)
        DBF_2 = 1/2 * (1 + eps - p2 - z1_w)
    elif z1_w <= 0 and z2_b <= 0: # reg N (no search)
        DBF_2 = 1/2 * (1 + eps - p2)
    
    D2_F = mu * DWF_2 + (1-mu) * DBF_2
    return D2_F

def D2_R(p1, eps, s, mu):
    z1_b, z1_w = z1(p1, eps, s)

    # Firm 1 is preferred:
    if z1_b > 0: # reg A or AN (active search after bad match)
        DWR_2 = 1/2 * (2*z1_b*(p1 - eps) + z1_b**2/2)
    else: # reg NA or N (no search after bad match)
        DWR_2 = 0

    # Firm 2 is preferred:
    if z1_w > 0: # reg A or NA (active search after good match)
        DBR_2 = 1/2 * (2*z1_w*p1 + z1_w**2/2)
    else: # reg AN or N (no search after good match)
        DBR_2 = 0
    
    D2_R = mu * DWR_2 + (1-mu) * DBR_2
    return D2_R

def D2(p1, p2, eps, s, mu):
    D2F = D2_F(p1, p2, eps, s, mu)
    D2R = D2_R(p1, eps, s, mu)
    
    return D2F + D2R

def profit1(p1, p2, eps, s, mu):
    return p1 * D1(p1, p2, eps, s, mu)

def profit2(p1, p2, eps, s, mu):
    return p2 * D2(p1, p2, eps, s, mu)

def BR1(p2, eps, s, mu, n=200):
    grid = np.linspace(0, 1+eps, n)
    profits = [profit1(p, p2, eps, s, mu) for p in grid]
    p0 = grid[np.argmax(profits)]

    res = minimize_scalar(
        lambda p: -profit1(p, p2, eps, s, mu),
        bounds = (max(0, p0-0.1), min(1+eps, p0+0.1)),
        method='bounded')
    return res.x

def BR2(p1, eps, s, mu, n=200):
    grid = np.linspace(0, 1+eps, n)
    profits = [profit1(p, p1, eps, s, mu) for p in grid]
    p0 = grid[np.argmax(profits)]

    res = minimize_scalar(
        lambda p: -profit2(p1, p, eps, s, mu),
        bounds = (max(0, p0-0.1), min(1+eps, p0+0.1)),
        method='bounded')
    return res.x

def solve_equilibrium(eps, s, mu, p1_init=0.5, p2_init=0.5, tol=1e-6, max_iter=500):
    
    p1, p2 = p1_init, p2_init
    
    for i in range(max_iter):
        p1_new = BR1(p2, eps, s, mu)
        p2_new = BR2(p1_new, eps, s, mu)
        
        if max(abs(p1_new - p1), abs(p2_new - p2)) < tol:
            return p1_new, p2_new, True
        
        p1, p2 = p1_new, p2_new
    
    return p1, p2, False

def consumer_surplus(p1, p2, eps, s, mu):
    
    EU_A = 0.5 * EU1_B(p1, p2, eps, s) + 0.5 * EU2_W(p1, p2, eps, s)
    
    EU_B = 0.5 * EU1_W(p1, p2, eps, s) + 0.5 * EU2_B(p1, p2, eps, s)

    return mu * EU_A + (1-mu) * EU_B

def producer_surplus(p1, p2, eps, s, mu):
    return profit1(p1, p2, eps, s, mu) + \
           profit2(p1, p2, eps, s, mu)

def total_welfare(p1, p2, eps, s, mu):
    return consumer_surplus(p1, p2, eps, s, mu) + \
           producer_surplus(p1, p2, eps, s, mu)