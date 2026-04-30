# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

def z_b(eps, s, p):
    z_b = 1 + eps - p - np.sqrt(2*s)
    return z_b

def z_w(s, p):
    z_w = 1 - p - np.sqrt(2*s)
    return z_w

def EU_num(eps, s, p):
    zw = z_w(s, p)
    zb = z_b(eps, s, p)
    EU_B = zw*(zw - eps + p) + ((1 + eps - p)**2 - zw**2)/2
    EU_W = zb * (zb + p) + ((1-p)**2-zb**2)/2
    return EU_B, EU_W

def EU_an(eps, s, p):
    EU_B = 1 + 1/2 * eps**2 + s - p - np.sqrt(2*s)*(1-eps)
    EU_W = 1 + 1/2 * eps**2 + s - p + eps - np.sqrt(2*s)*(1+eps)
    return EU_B, EU_W

def theta_num(eps, s, p):
    EUB, EUW = EU_num(eps, s, p)
    theta = (EUB - EUW)/2
    return theta

def theta_an(eps, s):
    theta = eps*np.sqrt(2*s) - eps/2
    return np.clip(theta,0,1)

def D(eps, s, p):
    zb = z_b(eps, s, p)
    zw = z_w(s, p)
    th = theta_an(eps, s)

    D_B = 1/2*(1 + th) * (1 + eps - p - zw*(1-2*p-1/2*zw))
    D_W = 1/2*(1 - th) * ((1 + eps - p)*(p + zb) - 1/2*zb**2)
    D = 1/2 * (D_B + D_W)
    return D

def profit(eps, s, p):
    Demand = D(eps, s, p)
    return p*Demand

def price(eps, s):
    obj = lambda p: -profit(eps, s, p)
    res = minimize_scalar(obj, bounds=(0.0, 1+eps), method='bounded')
    return res.x

def equilibrium_path_s_eps(eps_grid, s_grid):
    results = []
    for eps in eps_grid:
        for s in s_grid:
            p = price(eps, s)

            # compute objects
            the = theta_an(eps, s)

            results.append({"eps": eps, "s": s, "p1": p, "p2": p, "theta": the})
    return results


def plot_colorblock(df, eps_grid, s_grid):
    p1 = df.pivot(index='s', columns='eps', values='p1')
    p2 = df.pivot(index='s', columns='eps', values='p2')
    theta = df.pivot(index='s', columns='eps', values='theta')

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].contourf(eps_grid, s_grid, p1.values)
    ax[0].set_title("Price p1")
    ax[0].set_xlabel("epsilon")
    ax[0].set_ylabel("search cost s")
    ax[1].contourf(eps_grid, s_grid, p2.values)
    ax[1].set_title("Price p2")
    ax[1].set_xlabel("epsilon")
    ax[1].set_ylabel("search cost s")
    ax[2].contourf(eps_grid, s_grid, theta.values)
    ax[2].set_title("Disclosure share (theta*)")
    ax[2].set_xlabel("epsilon")
    ax[2].set_ylabel("search cost s")
    fig.colorbar(ax[0].collections[0], ax=ax[0])
    fig.colorbar(ax[1].collections[0], ax=ax[1])
    fig.colorbar(ax[2].collections[0], ax=ax[2])
    plt.tight_layout()
    plt.show()
