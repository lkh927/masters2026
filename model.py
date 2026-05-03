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
# TYPE A (prefers firm 1)
# TYPE B (prefers firm 2)
def z1(p1, eps, s):
    '''Reservation value for consumeres who visit firm 2 first. Defines the threshold match value for
    which consumers are indifferent between stopping search at firm 2 and paying s to search on to firm 1
        p1: Price charged by firm 1.
        eps: Preference shifter inducing the natural preference for firm 1.
        s: Search cost.'''
    z1_A = 1 + eps - p1 - np.sqrt(2*s)
    z1_B = 1 - p1 - np.sqrt(2*s)
    return max(z1_A,0), max(z1_B, 0)

def z2(p2, eps, s):
    '''Reservation value for consumeres who visit firm 1 first. Defines the threshold match value for
    which consumers are indifferent between stopping search at firm 1 and paying s to search on to firm 2
        p2: Price charged by firm 2.
        eps: Preference shifter inducing the natural preference for firm 2.
        s: Search cost.'''
    z2_A = 1 - p2 - np.sqrt(2*s)
    z2_B = 1 + eps - p2 - np.sqrt(2*s)
    return max(z2_A,0), max(z2_B,0)

# EXPECTED UTILITIES AND DISCLOSURE CUTOFF #
def EU1_A(p1, p2, eps, s):
    '''Expected utility of visiting firm 1 first, given that the first visit is free.
    EU1 = E[max{u1, z2, 0}]. Here for type A, who prefers firm 1.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    z_2, _ = z2(p2, eps, s)
    EU1 = z_2*(z_2 - eps + p1) + ((1 + eps - p1)**2 - z_2**2)/2
    return EU1

def EU1_B(p1, p2, eps, s):
    '''Expected utility of visiting firm 1 first, given that the first visit is free.
    EU1 = E[max{u1, z2, 0}]. Here for type B, who prefers firm 2.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    _, z_2 = z2(p2, eps, s)
    EU1 = z_2*(z_2 + p1) + ((1 - p1)**2 - z_2**2)/2
    return EU1

def EU2_A(p1, p2, eps, s):
    '''Expected utility of visiting firm 2 first, given that the first visit is free.
    EU2 = E[max{u2, z1, 0}]. Visiting firm 2 first happens with probability 0 if consumers decide to disclose
    and with probaiblity 0 if they do not. Here for type A.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost.'''
    z_1, _ = z1(p1, eps, s)
    EU2 = z_1*(z_1 + p2) + ((1-p2)**2 - z_1**2)/2
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
    EU2 = z_1*(z_1 - eps + p2) + ((1 + eps - p2)**2 - z_1**2)/2
    return EU2

def theta_star_A(p1, p2, eps, s, gamma, sigma):
    '''The disclosure cutoff for consumers who choose to disclose their preference for firm 1. 
    Disclosing this information comes at cost theta_i, which is the consumers' individual type, 
    capturing their privacy preferences. Here for type A.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost
    gamma: The share of naïve consumers, if any
    sigma: the scale parameter of the distribution of consumer types (uniform on [0,sigma])'''
    theta_star = gamma + (1-gamma)*((EU1_A(p1, p2, eps, s) - EU2_A(p1, p2, eps, s))/2)
    theta = theta_star/sigma
    return np.clip(theta, 0, 1)

def theta_star_B(p1, p2, eps, s, gamma, sigma):
    '''The disclosure cutoff for consumers who choose to disclose their preference for firm 1. 
    Disclosing this information comes at cost theta_i, which is the consumers' individual type, 
    capturing their privacy preferences. Here for type B.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost
    gamma: The share of naïve consumers, if any.
    sigma: the scale parameter of the distribution of consumer types (uniform on [0,sigma])'''
    theta_star = gamma + (1-gamma)*((EU2_B(p1, p2, eps, s) - EU1_B(p1, p2, eps, s))/2)
    theta = theta_star/sigma
    return np.clip(theta, 0, 1)

def Theta_star(p1, p2, eps, s, gamma, mu, sigma):
    '''The overall disclosure cutoff, as a weighted average of the disclosure cutoffs of type A and type B consumers.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost
    gamma: The share of naïve consumers, if any.
    mu: The share of type A consumers.
    sigma: the scale parameter of the distribution of consumer types (uniform on [0,sigma])'''
    Theta = mu * theta_star_A(p1, p2, eps, s, gamma, sigma) + (1-mu) * theta_star_B(p1, p2, eps, s, gamma, sigma)
    return Theta

# RANKING PROBABILITIES #
def ranking_probs(p1, p2, eps, s, gamma, mu, sigma):
    thetaA = theta_star_A(p1, p2, eps, s, gamma, sigma)
    thetaB = theta_star_B(p1, p2, eps, s, gamma, sigma)
    
    non_disclosure = mu*(1-thetaA) + (1-mu)*(1-thetaB)
    
    pi1_first = mu*thetaA + 0.5 * non_disclosure
    pi2_first = (1-mu)*thetaB + 0.5 * non_disclosure
    
    return pi1_first, pi2_first

# DEMAND #
# interior case relies on positive reservation values and 0 is in [eps-p_1, 1+eps-p_1] and [-p_2, 1-p_2]
def D1(p1, p2, eps, s, gamma, mu, sigma):
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
    gamma: The share of naïve consumers, if any.
    mu: The share of type A consumers.
    sigma: the scale parameter of the distribution of consumer types (uniform on [0,sigma])'''
    thetastar_A = theta_star_A(p1, p2, eps, s, gamma, sigma)
    thetastar_B = theta_star_B(p1, p2, eps, s, gamma, sigma)
    z1_A, z1_B = z1(p1, eps, s)
    z2_A, z2_B = z2(p2, eps, s)

    D1F_A = (1+thetastar_A)/2 *(1 + eps - p1 - z2_A) \
        + (1-thetastar_A)/2 * ((p2+z1_A)*(1 + eps - p1) - z1_A**2/2)
    D1R_A = (1+thetastar_A)/2 * (z2_A * (2*p2 + z2_A/2))
    D1_A = D1F_A + D1R_A

    D1F_B = (1-thetastar_B)/2 * (1 - z2_B - p1) \
        + (1+thetastar_B)/2 * ((p2 - eps + z1_B)*(1 - p1) - z1_B**2/2)
    D1R_B =  (1-thetastar_B)/2 * (2*z2_B*(p2 - eps) + z2_B**2/2)
    D1_B = D1F_B + D1R_B

    D1 = mu * D1_A + (1-mu) * D1_B
    return D1

def D2(p1, p2, eps, s, gamma, mu, sigma):
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
    gamma: The share of naïve consumers, if any.
    mu: The share of type A consumers.
    sigma: the scale parameter of the distribution of consumer types (uniform on [0,sigma])'''
    thetastar_A = theta_star_A(p1, p2, eps, s, gamma, sigma)
    thetastar_B = theta_star_B(p1, p2, eps, s, gamma, sigma)
    z1_A, z1_B = z1(p1, eps, s)
    z2_A, z2_B = z2(p2, eps, s)

    D2F_A = (1-thetastar_A)/2 * (1 - z1_A - p2) \
    + (1+thetastar_A)/2 * ((p1 - eps + z2_A)*(1 - p2) - z2_A**2/2)
    D2R_A = (1-thetastar_A)/2 * (2*z1_A*(p1 - eps) + z1_A**2/2)
    D2_A = D2F_A + D2R_A

    D2F_B = (1+thetastar_B)/2 * (1 + eps - p2 - z1_B) \
        + (1-thetastar_B)/2 * ((p1+z2_B)*(1 + eps - p2) - z2_B**2/2)
    D2R_B = (1+thetastar_B)/2 * (z1_B * (2*p1 + z1_B/2))
    D2_B = D2F_B + D2R_B

    D2 = mu * D2_A + (1-mu) * D2_B
    return D2

# FIRM PROFITS #
def profit1(p1, p2, eps, s, gamma, mu, sigma):
    return p1 * D1(p1, p2, eps, s, gamma, mu, sigma)

def profit2(p1, p2, eps, s, gamma, mu, sigma):
    return p2 * D2(p1, p2, eps, s, gamma, mu, sigma)

# BEST RESPONSE FUNCTIONS #
def BR1(p2, eps, s, gamma, mu, sigma):
    obj = lambda p1: -profit1(p1, p2, eps, s, gamma, mu, sigma)
    res = minimize_scalar(obj, bounds=(0, 1+eps), method='bounded')
    return res.x

def BR2(p1, eps, s, gamma, mu, sigma):
    obj = lambda p2: -profit2(p1, p2, eps, s, gamma, mu, sigma)
    res = minimize_scalar(obj, bounds=(0, 1+eps), method='bounded')
    return res.x


# EQUILIBRIUM SOLVER #
def solve_equilibrium(eps, s, gamma, mu, sigma, p1_init=0.5, p2_init=0.5, tol=1e-6, max_iter=500):
    
    p1, p2 = p1_init, p2_init
    
    for i in range(max_iter):
        p1_new = BR1(p2, eps, s, gamma, mu, sigma)
        p2_new = BR2(p1_new, eps, s, gamma, mu, sigma)
        
        if max(abs(p1_new - p1), abs(p2_new - p2)) < tol:
            return p1_new, p2_new, True
        
        p1, p2 = p1_new, p2_new
    
    return p1, p2, False

def check_interior(p1, p2, eps, s, gamma, mu, sigma):
    z1_A, z1_B = z1(p1, eps, s)
    z2_A, z2_B= z2(p2, eps, s)
    theta = Theta_star(p1, p2, eps, s, gamma, mu, sigma)
    
    cond_z = (z1_A > 0) and (z2_A > 0) and (z1_B > 0) and (z2_B > 0)
    cond_theta = (theta > 0) and (theta < 1)
    
    cond_support1 = (eps - p1 <= 0) and (1 + eps - p1 >= 0)
    cond_support2 = (-p2 <= 0) and (1 - p2 >= 0)
    
    return {
        "z_positive": cond_z,
        "theta_interior": cond_theta,
        "support_ok": cond_support1 and cond_support2,
        "interior": cond_z and cond_theta and cond_support1 and cond_support2
    }

def compute_equilibrium_path(eps, s, gamma_grid, mu, sigma):
    p1_list, p2_list, theta_list = [], [], []
    pi1_list, pi2_list = [], []
    CS_list, PS_list, W_list = [], [], []
    
    # warm start
    p1_init, p2_init = 0.5, 0.5
    
    for gamma in gamma_grid:
        p1, p2, converged = solve_equilibrium(
            eps, s, gamma, mu, sigma, 
            p1_init=p1_init,
            p2_init=p2_init
        )
        
        # update warm start
        p1_init, p2_init = p1, p2
        
        theta = Theta_star(p1, p2, eps, s, gamma, mu, sigma)

        pi1 = profit1(p1, p2, eps, s, gamma, mu, sigma)
        pi2 = profit2(p1, p2, eps, s, gamma, mu, sigma)

        CS = consumer_surplus(p1, p2, eps, s, gamma, mu, sigma)
        PS = producer_surplus(p1, p2, eps, s, gamma, mu, sigma)
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

def consumer_surplus(p1, p2, eps, s, gamma, mu, sigma):
    
    thetaA = theta_star_A(p1, p2, eps, s, gamma, sigma)
    thetaB = theta_star_B(p1, p2, eps, s, gamma, sigma)
    
    # TYPE A ranking
    pi1_A = thetaA + 0.5*(1-thetaA)
    pi2_A = 0.5*(1-thetaA)
    
    # TYPE B ranking
    pi2_B = thetaB + 0.5*(1-thetaB)
    pi1_B = 0.5*(1-thetaB)
    
    # Expected utility by type
    EU_A = pi1_A * EU1_A(p1, p2, eps, s) + pi2_A * EU2_A(p1, p2, eps, s)
    EU_B = pi1_B * EU1_B(p1, p2, eps, s) + pi2_B * EU2_B(p1, p2, eps, s)
    
    # Privacy costs (uniform distribution)
    cost_A = thetaA**2 / 2
    cost_B = thetaB**2 / 2
    
    return mu * (EU_A - cost_A) + (1-mu) * (EU_B - cost_B)

def producer_surplus(p1, p2, eps, s, gamma, mu, sigma):
    return profit1(p1, p2, eps, s, gamma, mu, sigma) + profit2(p1, p2, eps, s, gamma, mu, sigma)

def total_welfare(p1, p2, eps, s, gamma, mu, sigma):
    return consumer_surplus(p1, p2, eps, s, gamma, mu, sigma) + \
           producer_surplus(p1, p2, eps, s, gamma, mu, sigma)

def equilibrium_path_s_eps(eps_grid, s_grid, gamma, mu, sigma, prev_p1, prev_p2):
    results = []
    for eps in eps_grid:
        for s in s_grid:
            try:
                p1, p2, converged = solve_equilibrium(eps, s, gamma, mu, sigma, p1_init=prev_p1, p2_init=prev_p2)

                # update warm start
                prev_p1, prev_p2 = p1, p2

                # compute objects
                Theta = Theta_star(p1, p2, eps, s, gamma, mu, sigma)
                interior = check_interior(p1, p2, eps, s, gamma, mu, sigma)['interior']

                results.append({"eps": eps, "s": s, "p1": p1, "p2": p2, "Theta": Theta,
                                 "interior": interior, "converged": converged})
                
            except Exception as e:
                results.append({"eps": eps, "s": s, "p1": np.nan, "p2": np.nan, "Theta": np.nan,
                                 "interior": False, "converged": False})
        
    return results

def plot_colorblock(df, eps_grid, s_grid):
    p1 = df.pivot(index='s', columns='eps', values='p1')
    p2 = df.pivot(index='s', columns='eps', values='p2')
    theta = df.pivot(index='s', columns='eps', values='Theta')

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

def plot_price_curves(eps_grid, s_values, gamma, mu, p1_init, p2_init):
    fig, axes = plt.subplots(1, 3, figsize=(18,5), sharey=True)
    for i, s in enumerate(s_values):
        ax = axes[i]
        
        p1_list, p2_list = [], []
        
        for eps in eps_grid:
            p1, p2, _ = solve_equilibrium(eps, s, gamma, mu, p1_init, p2_init)

            p1_init, p2_init = p1, p2

            p1_list.append(p1)
            p2_list.append(p2)

        ax.plot(eps_grid, p1_list, label='p1')
        ax.plot(eps_grid, p2_list, label='p2')

        ax.set_title(f"Search cost = {s}")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("price")
        ax.set_ylim(0.3, 0.7)
    
        ax.legend()
    
    plt.tight_layout
    plt.show()

def plot_welfare_comparison(df):
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
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title("Model comparison")
    plt.legend()

    plt.show()
