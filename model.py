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

def theta_star(p1, p2, eps, s, gamma):
    '''The disclosure cutoff for consumers who choose to disclose their preference for firm 1. 
    Disclosing this information comes at cost theta_i, which is the consumers' individual type, 
    capturing their privacy preferences. Consumers with theta_i ≤ theta_star disclose, while consumers
    with theta_i > theta_star do not.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost
    gamma: The share of naïve consumers, if any.'''
    theta = gamma + (1-gamma)*((EU1(p1, p2, eps, s) - EU2(p1, p2, eps, s))/2)
    return np.clip(theta, 0, 1)

# DEMAND #
# interior case relies on positive reservation values and 0 is in [eps-p_1, 1+eps-p_1] and [-p_2, 1-p_2]
def D1(p1, p2, eps, s, gamma):
    '''Demand for firm 1: Fresh demand + return demand.
    Fresh demand: the demand from consumers who buy from firm 1 immediately upon visiting it - so
    the sum of consumers who buy immediately when seeing firm 1 first and those who search to firm 1 and then buy
    immediately when seeing firm 2 first.
    Return demand: the demand from consumers who buy from firm 1  after searching to firm 2 first and then returning 
    to firm 1.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.
    gamma: The share of naïve consumers, if any.'''
    thetastar = theta_star(p1, p2, eps, s, gamma)
    z_1 = z1(p1, eps, s)
    z_2 = z2(p2, s)
    D1F = (1+thetastar)/2 *(1 + eps - p1 - z_2) + (1-thetastar)/2 * ((p2+z_1)*(1 + eps - p1) - z_1**2/2)
    D1R = (1+thetastar)/2 * (z_2 * (2*p2 + z_2/2))
    D1 = D1F + D1R
    return D1

def D2(p1, p2, eps, s, gamma):
    '''Demand for firm 2: Fresh demand + return demand.
    Fresh demand: the demand from consumers who buy from firm 2 immediately upon visiting it - so
    the sum of consumers who buy immediately when seeing firm 2 first and those who search to firm 2 and then buy
    immediately when seeing firm 1 first.
    Return demand: the demand from consumers who buy from firm 2  after searching to firm 1 first and then returning 
    to firm 2.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.
    gamma: The share of naïve consumers, if any.'''
    thetastar = theta_star(p1, p2, eps, s, gamma)
    z_1 = z1(p1, eps, s)
    z_2 = z2(p2, s)
    D2F = (1-thetastar)/2 * (1 - z_1 - p2) + (1+thetastar)/2 * ((p1 - eps + z_2) * (1 - p2) - z_2**2/2)
    D2R = (1-thetastar)/2 * (2*z_1 * (p1 - eps) + z_1**2/2)
    D2 = D2F + D2R
    return D2

# FIRM PROFITS #
def profit1(p1, p2, eps, s, gamma):
    return p1 * D1(p1, p2, eps, s, gamma)

def profit2(p1, p2, eps, s, gamma):
    return p2 * D2(p1, p2, eps, s, gamma)

# BEST RESPONSE FUNCTIONS #
def BR1(p2, eps, s, gamma):
    obj = lambda p1: -profit1(p1, p2, eps, s, gamma)
    res = minimize_scalar(obj, bounds=(0, 1+eps), method='bounded')
    return res.x

def BR2(p1, eps, s, gamma):
    obj = lambda p2: -profit2(p1, p2, eps, s, gamma)
    res = minimize_scalar(obj, bounds=(0, 1+eps), method='bounded')
    return res.x


# EQUILIBRIUM SOLVER #
def solve_equilibrium(eps, s, gamma, p1_init=0.5, p2_init=0.5, tol=1e-6, max_iter=500):
    
    p1, p2 = p1_init, p2_init
    
    for i in range(max_iter):
        p1_new = BR1(p2, eps, s, gamma)
        p2_new = BR2(p1_new, eps, s, gamma)
        
        if max(abs(p1_new - p1), abs(p2_new - p2)) < tol:
            return p1_new, p2_new, True
        
        p1, p2 = p1_new, p2_new
    
    return p1, p2, False

def check_interior(p1, p2, eps, s, gamma):
    z_1 = z1(p1, eps, s)
    z_2 = z2(p2, s)
    theta = theta_star(p1, p2, eps, s, gamma)
    
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

def compute_equilibrium_path(eps, s, gamma_grid):
    p1_list, p2_list, theta_list = [], [], []
    pi1_list, pi2_list = [], []
    CS_list, PS_list, W_list = [], [], []
    
    # warm start
    p1_init, p2_init = 0.5, 0.5
    
    for gamma in gamma_grid:
        p1, p2, converged = solve_equilibrium(
            eps, s, gamma,
            p1_init=p1_init,
            p2_init=p2_init
        )
        
        # update warm start
        p1_init, p2_init = p1, p2
        
        theta = theta_star(p1, p2, eps, s, gamma)

        pi1 = profit1(p1, p2, eps, s, gamma)
        pi2 = profit2(p1, p2, eps, s, gamma)

        CS = consumer_surplus(p1, p2, eps, s, gamma)
        PS = producer_surplus(p1, p2, eps, s, gamma)
        W  = CS + PS
        
        p1_list.append(p1)
        p2_list.append(p2)
        pi1_list.append(pi1)
        pi2_list.append(pi2)
        theta_list.append(theta)
        CS_list.append(CS)
        PS_list.append(PS)
        W_list.append(W)
    
    return np.array(p1_list), np.array(p2_list), np.array(pi1_list), np.array(pi2_list), np.array(theta_list), np.array(CS_list), np.array(PS_list), np.array(W_list)

# WELFARE OUTCOMES #
def consumer_surplus(p1, p2, eps, s, gamma):
    theta = theta_star(p1, p2, eps, s, gamma)
    
    EU_1 = EU1(p1, p2, eps, s)
    EU_2 = EU2(p1, p2, eps, s)
    
    # expected utility
    EU = (1 + theta)/2 * EU_1 + (1 - theta)/2 * EU_2
    
    # disclosure cost
    disclosure_cost = theta**2 / 2
    
    return EU - disclosure_cost

def producer_surplus(p1, p2, eps, s, gamma):
    return profit1(p1, p2, eps, s, gamma) + profit2(p1, p2, eps, s, gamma)

def total_welfare(p1, p2, eps, s, gamma):
    return consumer_surplus(p1, p2, eps, s, gamma) + \
           producer_surplus(p1, p2, eps, s, gamma)

def plot_comparison(df):

    metrics = ["CS", "PS", "W", "pi1", "pi2"]
    x = np.arange(len(metrics))
    width = 0.25

    plt.figure()

    for i, model in enumerate(df.index):
        values = [df.loc[model, m] for m in metrics]
        plt.bar(x + i*width, values, width, label=model)

    plt.xticks(x + width, metrics)
    plt.xlabel("Outcome")
    plt.ylabel("Value")
    plt.title("Model comparison")
    plt.legend()

    plt.show()