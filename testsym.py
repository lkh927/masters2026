# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

def theta(eps, s):
    theta = eps*np.sqrt(2*s) - eps
    return theta

def D(eps, s, p):
    th = theta(eps, s)
    D = 1/2 * ( (1 + th) * (eps + p - (3*p**2)/2 + p*np.sqrt(2*s)+s) \
                 +(1 - th) * (1/2 + eps + 1/2*eps - 2*np.sqrt(2*s) - 2*p*np.sqrt(2*s) - 1/2*p**2 - s) )
    return D

def profit(eps, s, p):
    Demand = D(eps, s, p)
    return p*Demand

def BR(eps, s):
    obj = lambda p: -profit(eps, s, p)
    res = minimize_scalar(obj, bounds=(0.0, 1+eps), method='bounded')
    return res.x