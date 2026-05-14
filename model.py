# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})
BLUE_CMAP = plt.cm.Blues

# RESERVATION VALUES #
# TYPE A (prefers firm 1)
# TYPE B (prefers firm 2)
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

def theta_star_1(p1, p2, eps, s, gamma, sigma):
    '''The disclosure cutoff for consumers who choose to disclose their preference for firm 1. 
    Disclosing this information comes at cost theta_i, which is the consumers' individual type, 
    capturing their privacy preferences. Here for type A.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost
    gamma: The share of naïve consumers, if any
    sigma: the scale parameter of the distribution of consumer types (uniform on [0,sigma])'''
    theta_star = gamma + (1-gamma)*((EU1_B(p1, p2, eps, s) - EU2_W(p1, p2, eps, s))/2)
    theta = theta_star/sigma
    return np.clip(theta, 0, 1)

def theta_star_2(p1, p2, eps, s, gamma, sigma):
    '''The disclosure cutoff for consumers who choose to disclose their preference for firm 1. 
    Disclosing this information comes at cost theta_i, which is the consumers' individual type, 
    capturing their privacy preferences. Here for type B.
    p1: Price charged by firm 1.
    p2: Price charged by firm 2.
    eps: Preference shifter inducing the natural preference for firm 1.
    s: Search cost
    gamma: The share of naïve consumers, if any.
    sigma: the scale parameter of the distribution of consumer types (uniform on [0,sigma])'''
    theta_star = gamma + (1-gamma)*((EU2_B(p1, p2, eps, s) - EU1_W(p1, p2, eps, s))/2)
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
    Theta = mu * theta_star_1(p1, p2, eps, s, gamma, sigma) + (1-mu) * theta_star_2(p1, p2, eps, s, gamma, sigma)
    return Theta

# RANKING PROBABILITIES #
def ranking_probs(p1, p2, eps, s, gamma, mu, sigma):
    theta1 = theta_star_1(p1, p2, eps, s, gamma, sigma)
    theta2 = theta_star_2(p1, p2, eps, s, gamma, sigma)
    
    non_disclosure = mu*(1-theta1) + (1-mu)*(1-theta2)
    
    R1_first = mu*theta1 + 0.5 * non_disclosure
    R2_first = (1-mu)*theta2 + 0.5 * non_disclosure

    return R1_first, R2_first

# DEMAND #
# interior case relies on positive reservation values and 0 is in [eps-p_1, 1+eps-p_1] and [-p_2, 1-p_2]

def D1_F(p1, p2, eps, s, gamma, mu, sigma):
    thetastar_1 = theta_star_1(p1, p2, eps, s, gamma, sigma)
    thetastar_2 = theta_star_2(p1, p2, eps, s, gamma, sigma)
    z1_b, z1_w = z1(p1, eps, s)
    z2_w, z2_b = z2(p2, eps, s)

    # Firm 1 is preferred: D1_F = DBF_1 (preferred firm, F=1)
    if z1_b > 0 and z2_w > 0: # reg A (active search)
        DBF_1 = (1+thetastar_1)/2 *(1 + eps - p1 - z2_w) \
        + (1-thetastar_1)/2 * ((p2+z1_b)*(1 + eps - p1) - z1_b**2/2)
    elif z1_b > 0 and z2_w <= 0: # reg AN (no search after good match)
        DBF_1 = (1+thetastar_1)/2 *(1 + eps - p1) \
        + (1-thetastar_1)/2 * ((p2+z1_b)*(1 + eps - p1) - z1_b**2/2)
    elif z1_b <= 0 and z2_w > 0: # reg NA (no search after bad match)
        DBF_1 = (1+thetastar_1)/2 *(1 + eps - p1 - z2_w) 
    elif z1_b <=0 and z2_w <= 0: # reg N (no search)
        DBF_1 = (1+thetastar_1)/2 *(1 + eps - p1)

    # Firm 2 is preferred: D1_F = DWF_1 (non-preferred firm, F=1)
    if z1_w > 0 and z2_b > 0: # reg A (active search)
        DWF_1 = (1-thetastar_2)/2 * (1 - z2_b - p1) \
        + (1+thetastar_2)/2 * ((p2 - eps + z1_w)*(1 - p1) - z1_w**2/2)
    elif z1_w <=0 and z2_b > 0: # reg AN (no search after good match)
        DWF_1 = (1-thetastar_2)/2 * (1 - z2_b - p1)
    elif z1_w > 0 and z2_b <= 0: # reg NA (no search after bad match)
        DWF_1 = (1-thetastar_2)/2 * (1 - p1) \
        + (1+thetastar_2)/2 * ((p2 - eps + z1_w)*(1 - p1) - z1_w**2/2)
    elif z1_w <= 0 and z2_b <= 0: # reg N (no search)
        DWF_1 = (1-thetastar_2)/2 * (1 - p1)

    D1_F = mu * DBF_1 + (1-mu) * DWF_1
    return D1_F

def D1_R(p1, p2, eps, s, gamma, mu, sigma):
    thetastar_1 = theta_star_1(p1, p2, eps, s, gamma, sigma)
    thetastar_2 = theta_star_2(p1, p2, eps, s, gamma, sigma)
    z2_w, z2_b = z2(p2, eps, s)

    # Firm 1 is preferred:
    if z2_w > 0: # reg A or AN (active search after good match)
        DBR_1 = (1+thetastar_1)/2 * (2*z2_w*p2 + z2_w**2/2)
    else: # reg NA or N (no search after good match)
        DBR_1 = 0
    
    # Firm 2 is preferred:
    if z2_b > 0: # reg A or AN (active search after bad match)
        DWR_1 = (1-thetastar_2)/2 * (2*z2_b*(p2 - eps) + z2_b**2/2)
    else: # reg NA or N (no search after bad match)
        DWR_1 = 0
    
    D1_R = mu * DBR_1 + (1-mu) * DWR_1
    return D1_R

def D1(p1, p2, eps, s, gamma, mu, sigma):
    D1F = D1_F(p1, p2, eps, s, gamma, mu, sigma)
    D1R = D1_R(p1, p2, eps, s, gamma, mu, sigma)
    return D1F + D1R

def D2_F(p1, p2, eps, s, gamma, mu, sigma):
    thetastar_1 = theta_star_1(p1, p2, eps, s, gamma, sigma)
    thetastar_2 = theta_star_2(p1, p2, eps, s, gamma, sigma)
    z1_b, z1_w = z1(p1, eps, s)
    z2_w, z2_b = z2(p2, eps, s)

    # Firm 1 is preferred: D2_F = DWF_2 (non-preferred firm, F=2)
    if z1_b > 0 and z2_w > 0: # reg A (active search)
        DWF_2 = (1-thetastar_1)/2 *(1 - z1_b - p2) \
        + (1+thetastar_1)/2 * ((p1 - eps + z2_w)*(1 - p2) - z2_w**2/2)
    elif z1_b > 0 and z2_w <= 0: # reg AN (no search after good match)
        DWF_2 = (1-thetastar_1)/2 *(1 - z1_b - p2)
    elif z1_b <= 0 and z2_w > 0: # reg NA (no search after bad match)
        DWF_2 = (1-thetastar_1)/2 *(1 - p2) \
        + (1+thetastar_1)/2 * ((p1 - eps + z2_w)*(1 - p2) - z2_w**2/2)
    elif z1_b <=0 and z2_w <= 0: # reg N (no search)
        DWF_2 = (1-thetastar_1)/2 * (1 - p2)
    
    # Firm 2 is prererred: D2_F = DBF_2 (preferred firm, F=2)
    if z1_w > 0 and z2_b > 0: # reg A (active search)
        DBF_2 = (1+thetastar_2)/2 * (1 + eps - p2 - z1_w) \
        + (1-thetastar_2)/2 * ((p1+z2_b)*(1 + eps - p2) - z2_b**2/2)
    elif z1_w <=0 and z2_b > 0: # reg AN (no search after good match)
        DBF_2 = (1+thetastar_2)/2 * (1 + eps - p2) \
        + (1-thetastar_2)/2 * ((p1+z2_b)*(1 + eps - p2) - z2_b**2/2)
    elif z1_w > 0 and z2_b <= 0: # reg NA (no search after bad match)
        DBF_2 = (1+thetastar_2)/2 * (1 + eps - p2 - z1_w)
    elif z1_w <= 0 and z2_b <= 0: # reg N (no search)
        DBF_2 = (1+thetastar_2)/2 * (1 + eps - p2)
    
    D2_F = mu * DWF_2 + (1-mu) * DBF_2
    return D2_F

def D2_R(p1, p2, eps, s, gamma, mu, sigma):
    thetastar_1 = theta_star_1(p1, p2, eps, s, gamma, sigma)
    thetastar_2 = theta_star_2(p1, p2, eps, s, gamma, sigma)
    z1_b, z1_w = z1(p1, eps, s)

    # Firm 1 is preferred:
    if z1_b > 0: # reg A or AN (active search after bad match)
        DWR_2 = (1-thetastar_1)/2 * (2*z1_b*(p1 - eps) + z1_b**2/2)
    else: # reg NA or N (no search after bad match)
        DWR_2 = 0

    # Firm 2 is preferred:
    if z1_w > 0: # reg A or NA (active search after good match)
        DBR_2 = (1+thetastar_2)/2 * (2*z1_w*p1 + z1_w**2/2)
    else: # reg AN or N (no search after good match)
        DBR_2 = 0
    
    D2_R = mu * DWR_2 + (1-mu) * DBR_2
    return D2_R

def D2(p1, p2, eps, s, gamma, mu, sigma):
    D2F = D2_F(p1, p2, eps, s, gamma, mu, sigma)
    D2R = D2_R(p1, p2, eps, s, gamma, mu, sigma)
    
    return D2F + D2R

# FIRM PROFITS #
def profit1(p1, p2, eps, s, gamma, mu, sigma):
    return p1 * D1(p1, p2, eps, s, gamma, mu, sigma)

def profit2(p1, p2, eps, s, gamma, mu, sigma):
    return p2 * D2(p1, p2, eps, s, gamma, mu, sigma)

# BEST RESPONSE FUNCTIONS #
def BR1(p2, eps, s, gamma, mu, sigma, n=200):
    grid = np.linspace(0, 1+eps, n)
    profits = [profit1(p, p2, eps, s, gamma, mu, sigma) for p in grid]
    p0 = grid[np.argmax(profits)]

    res = minimize_scalar(
        lambda p: -profit1(p, p2, eps, s, gamma, mu, sigma),
        bounds = (max(0, p0-0.1), min(1+eps, p0+0.1)),
        method='bounded')
    return res.x

def BR2(p1, eps, s, gamma, mu, sigma, n=200):
    grid = np.linspace(0, 1+eps, n)
    profits = [profit2(p1, p, eps, s, gamma, mu, sigma) for p in grid]
    p0 = grid[np.argmax(profits)]

    res = minimize_scalar(
        lambda p: -profit2(p1, p, eps, s, gamma, mu, sigma),
        bounds = (max(0, p0-0.1), min(1+eps, p0+0.1)),
        method='bounded')
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
    
    theta1 = theta_star_1(p1, p2, eps, s, gamma, sigma)
    theta2 = theta_star_2(p1, p2, eps, s, gamma, sigma)
    
    # TYPE A ranking
    pi1_B = theta1 + 0.5*(1-theta1)
    pi2_W = 0.5*(1-theta1)
    
    # TYPE B ranking
    pi2_B = theta2 + 0.5*(1-theta2)
    pi1_W = 0.5*(1-theta2)
    
    # Expected utility by type
    EU_A = pi1_B * EU1_B(p1, p2, eps, s) + pi2_W * EU2_W(p1, p2, eps, s)
    EU_B = pi1_W * EU1_W(p1, p2, eps, s) + pi2_B * EU2_B(p1, p2, eps, s)
    
    # Privacy costs (uniform distribution)
    cost_A = theta1**2 / 2
    cost_B = theta2**2 / 2
    
    return mu * (EU_A - cost_A) + (1-mu) * (EU_B - cost_B)

def producer_surplus(p1, p2, eps, s, gamma, mu, sigma):
    return profit1(p1, p2, eps, s, gamma, mu, sigma) + profit2(p1, p2, eps, s, gamma, mu, sigma)

def total_welfare(p1, p2, eps, s, gamma, mu, sigma):
    return consumer_surplus(p1, p2, eps, s, gamma, mu, sigma) + \
           producer_surplus(p1, p2, eps, s, gamma, mu, sigma)


##### EQUILIBRIUM CALIBRATION #####

def classify_regime(p1, p2, eps, s):
    z1_b, z1_w = z1(p1, eps, s)
    z2_w, z2_b = z2(p2, eps, s)
    
    # Use type A logic (sufficient due to symmetry between z1_b, z2_b and z1_w, z2_w)
    if z2_w > 0 and z1_b > 0:
        return "A"      # full search
    elif z2_w <= 0 and z1_b > 0:
        return "AN"     # no search after good match
    elif z2_w > 0 and z1_b <= 0:
        return "NA"     # no search after bad match
    else:
        return "N"      # no search at all


def equilibrium_path_s_eps(eps_grid, s_grid, gamma, mu, sigma, prev_p1=0.5, prev_p2=0.5):
    results = []

    for eps in eps_grid:
        for s in s_grid:
            try:
                p1, p2, converged = solve_equilibrium(eps, s, gamma, mu, sigma,
                    p1_init=prev_p1,p2_init=prev_p2)

                # warm start update
                prev_p1, prev_p2 = p1, p2

                # key objects
                Theta = Theta_star(p1, p2, eps, s, gamma, mu, sigma)
                interior = check_interior(p1, p2, eps, s, gamma, mu, sigma)["interior"]
                regime = classify_regime(p1, p2, eps, s)

                results.append({"eps": eps, "s": s, "p1": p1, "p2": p2, "Theta": Theta,
                                "regime": regime, "interior": interior,"converged": converged})

            except Exception:
                results.append({"eps": eps,"s": s,"p1": np.nan,"p2": np.nan,"Theta": np.nan,
                    "regime": "fail","interior": False,"converged": False})

    return pd.DataFrame(results)

def plot_colorblock(df, eps_grid, s_grid):
    p1 = df.pivot(index='s', columns='eps', values='p1')
    p2 = df.pivot(index='s', columns='eps', values='p2')
    theta = df.pivot(index='s', columns='eps', values='Theta')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    plots = [
        (p1, "Price $p_1$"),
        (p2, "Price $p_2$"),
        (theta, r"Disclosure share $\Theta^*$")
    ]

    for ax, (data, title) in zip(axes, plots):
        cont = ax.contourf(
            eps_grid,
            s_grid,
            data.values,
            levels=20,
            cmap=BLUE_CMAP
        )

        ax.set_title(title)
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel(r"$s$")

        # Clean grid
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Colorbar per subplot
        cbar = fig.colorbar(cont, ax=ax)
        cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.show()

def find_mixed_regions(df_regime):
    mixed = df_regime[df_regime["regime"].isin(["AN"])]
    return mixed[["eps", "s", "p1", "p2"]]

def plot_regime_map(df, eps_grid, s_grid):

    regime_dict = {"A": 0, "AN": 1, "NA": 2, "N": 3}

    Z = df.assign(regime_code=df["regime"].map(regime_dict)) \
          .pivot(index="s", columns="eps", values="regime_code")

    fig, ax = plt.subplots(figsize=(6,5))

    mesh = ax.pcolormesh(
        eps_grid,
        s_grid,
        Z.values,
        shading='nearest',
        cmap=BLUE_CMAP
    )

    cbar = plt.colorbar(mesh)
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(["A", "AN", "NA", "N"])

    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"search cost $s$")

    plt.show()

def plot_price_curves(eps_grid, s_values, gamma, mu, sigma, p1_init=0.5, p2_init=0.5):
    fig, axes = plt.subplots(1, 3,figsize=(18,5))

    for i, s in enumerate(s_values):
        ax = axes[i]

        p1_list, p2_list, theta_list = [], [], []
        regimes = []

        p1_curr, p2_curr = p1_init, p2_init
        for eps in eps_grid:

            p1, p2, _ = solve_equilibrium(eps, s, gamma, mu, sigma, p1_init=p1_curr, p2_init=p2_curr)
            theta = Theta_star(p1, p2, eps, s, gamma, mu, sigma)
            regime = classify_regime(p1, p2, eps, s)

            p1_curr, p2_curr = p1, p2

            p1_list.append(p1)
            p2_list.append(p2)
            theta_list.append(theta)
            regimes.append(regime)

        # --- prices ---
        ax.plot(eps_grid, p1_list, label=r"$p_1$", color="indigo", linewidth=2)
        ax.plot(eps_grid, p2_list, label=r"$p_2$", color="slateblue", linewidth=2)

        # regime boundaries
        for j in range(1, len(eps_grid)):
            if regimes[j] != regimes[j-1]:
                boundary = 0.5 * (eps_grid[j] + eps_grid[j-1])
                ax.axvline(boundary, color='black',linestyle='--', alpha=0.5)

        ax.set_title(rf"Search cost = {s}")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylim(0.2, 0.7)
        ax.grid(axis='y',linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel("Price")
        ax.tick_params(axis='y', labelleft=True)

        # aligned ticks
        price_ticks = np.arange(0.2, 0.71, 0.1)
        ax.set_yticks(price_ticks)

        # --- disclosure axis ---
        ax2 = ax.twinx()
        ax2.plot(eps_grid, theta_list, color="cornflowerblue", linestyle='--', linewidth=2.5, label=r"$\Theta^*$")
        ax2.set_ylim(0, 0.5)
        ax2.set_yticks(price_ticks - 0.2)
        ax2.spines['top'].set_visible(False)

        ax2.set_ylabel(r"Disclosure share $\Theta^*$")

    # combined legend
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    fig.legend(lines1 + lines2,labels1 + labels2,loc='upper center',ncol=3,frameon=False)
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()

def plot_price_curves_s(s_grid, eps_values, gamma, mu, sigma, p1_init=0.5, p2_init=0.5):
    fig, axes = plt.subplots(1, len(eps_values), figsize=(18,5), sharey=True)

    for i, eps in enumerate(eps_values):
        ax = axes[i]

        p1_list, p2_list, theta_list = [], [], []
        regimes = []        
        
        # reset warm start for each epsilon
        p1_curr, p2_curr = p1_init, p2_init
        for s in s_grid:
            p1, p2, _ = solve_equilibrium(eps, s, gamma, mu, sigma,
                p1_init=p1_curr, p2_init=p2_curr)
            theta = Theta_star(p1, p2, eps, s, gamma, mu, sigma)
            regime = classify_regime(p1, p2, eps, s)

            # update warm start along s-dimension
            p1_curr, p2_curr = p1, p2

            p1_list.append(p1)
            p2_list.append(p2)
            theta_list.append(theta)
            regimes.append(regime)

        ax.plot(s_grid, p1_list, label=r"$p_1$", color="indigo", linewidth=2)
        ax.plot(s_grid, p2_list, label=r"$p_2$", color="slateblue", linewidth=2)

        for j in range(1, len(s_grid)):
            if regimes[j] != regimes[j-1]:
                boundary = 0.5 * (s_grid[j] + s_grid[j-1])
                ax.axvline(boundary, color='black', linestyle='--', alpha=0.5)

        ax.set_title(rf"$\epsilon = {eps}$")
        ax.set_xlabel(r"$s$")
        ax.set_ylim(0.3, 0.7)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel("Price")
        ax.tick_params(axis='y', labelleft=True)

        # aligned ticks
        price_ticks = np.arange(0.2, 0.71, 0.1)
        ax.set_yticks(price_ticks)
    
    # disclosure axiz
    ax2 = ax.twinx()
    ax2.plot(s_grid, theta_list, color="cornflowerblue", linestyle='--', linewidth=2.5, label=r"$\Theta^*$")
    ax2.set_ylim(0, 0.5)
    ax2.set_yticks(price_ticks-0.2)
    ax2.spines['top'].set_visible(False)

    ax2.set_ylabel(r"Disclosure share $\Theta^*$")

    # combined legend
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=3, frameon=False)

    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()


###### NO SHARING VERSION OF THE MODEL #######
def D1_F_NS(p1, p2, eps, s, mu):
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


def D1_R_NS(p2, eps, s, mu):
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

def D1_NS(p1, p2, eps, s, mu):
    D1F = D1_F_NS(p1, p2, eps, s, mu)
    D1R = D1_R_NS(p2, eps, s, mu)
    return D1F + D1R


def D2_F_NS(p1, p2, eps, s, mu):
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

def D2_R_NS(p1, eps, s, mu):
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

def D2_NS(p1, p2, eps, s, mu):
    D2F = D2_F_NS(p1, p2, eps, s, mu)
    D2R = D2_R_NS(p1, eps, s, mu)
    
    return D2F + D2R

def profit1_NS(p1, p2, eps, s, mu):
    return p1 * D1_NS(p1, p2, eps, s, mu)

def profit2_NS(p1, p2, eps, s, mu):
    return p2 * D2_NS(p1, p2, eps, s, mu)

def BR1_NS(p2, eps, s, mu, n=200):
    grid = np.linspace(0, 1+eps, n)
    profits = [profit1_NS(p, p2, eps, s, mu) for p in grid]
    p0 = grid[np.argmax(profits)]

    res = minimize_scalar(
        lambda p: -profit1_NS(p, p2, eps, s, mu),
        bounds = (max(0, p0-0.1), min(1+eps, p0+0.1)),
        method='bounded')
    return res.x

def BR2_NS(p1, eps, s, mu, n=200):
    grid = np.linspace(0, 1+eps, n)
    profits = [profit2_NS(p1, p, eps, s, mu) for p in grid]
    p0 = grid[np.argmax(profits)]

    res = minimize_scalar(
        lambda p: -profit2_NS(p1, p, eps, s, mu),
        bounds = (max(0, p0-0.1), min(1+eps, p0+0.1)),
        method='bounded')
    return res.x

def solve_equilibrium_NS(eps, s, mu, p1_init=0.5, p2_init=0.5, tol=1e-6, max_iter=500):
    
    p1, p2 = p1_init, p2_init
    
    for i in range(max_iter):
        p1_new = BR1_NS(p2, eps, s, mu)
        p2_new = BR2_NS(p1_new, eps, s, mu)
        
        if max(abs(p1_new - p1), abs(p2_new - p2)) < tol:
            return p1_new, p2_new, True
        
        p1, p2 = p1_new, p2_new
    
    return p1, p2, False

def consumer_surplus_NS(p1, p2, eps, s, mu):
    
    EU_A = 0.5 * EU1_B(p1, p2, eps, s) + 0.5 * EU2_W(p1, p2, eps, s)
    
    EU_B = 0.5 * EU1_W(p1, p2, eps, s) + 0.5 * EU2_B(p1, p2, eps, s)

    return mu * EU_A + (1-mu) * EU_B

def producer_surplus_NS(p1, p2, eps, s, mu):
    return profit1_NS(p1, p2, eps, s, mu) + \
           profit2_NS(p1, p2, eps, s, mu)

def total_welfare_NS(p1, p2, eps, s, mu):
    return consumer_surplus_NS(p1, p2, eps, s, mu) + \
           producer_surplus_NS(p1, p2, eps, s, mu)

def compare_outcomes_benchmark(eps, s, mu, sigma):

    results = {}
    # --- No sharing ---
    p1_ns, p2_ns, _ = solve_equilibrium_NS(eps, s, mu)
    pi1_ns = profit1_NS(p1_ns, p2_ns, eps, s, mu)
    pi2_ns = profit2_NS(p1_ns, p2_ns, eps, s, mu)
    regime_ns = classify_regime(p1_ns, p2_ns, eps, s)

    results["Benchmark: No Disclosure"] = {
        "p1": p1_ns,
        "p2": p2_ns,
        "pi1": pi1_ns,
        "pi2": pi2_ns,
        "CS": consumer_surplus_NS(p1_ns, p2_ns, eps, s, mu),
        "PS": pi1_ns + pi2_ns,
        "W": total_welfare_NS(p1_ns, p2_ns, eps, s, mu),
        "Theta": 0,
        "regime": regime_ns
    }

    # --- Sharing, rational ---
    p1_r, p2_r, _ = solve_equilibrium(eps, s, gamma=0, mu=mu, sigma=sigma)
    pi1_r = profit1(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma)
    pi2_r = profit2(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma)

    Theta_r = Theta_star(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma)
    regime_r = classify_regime(p1_r, p2_r, eps, s)

    results["Full Model with Disclosure"] = {
        "p1": p1_r,
        "p2": p2_r,
        "pi1": pi1_r,
        "pi2": pi2_r,
        "CS": consumer_surplus(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma),
        "PS": pi1_r + pi2_r,
        "W": total_welfare(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma),
        "Theta": Theta_r,
        "regime": regime_r
    }
    return pd.DataFrame(results).T

def compare_outcomes(eps, s, mu, sigma):

    results = {}

    # --- No sharing ---
    p1_ns, p2_ns, _ = solve_equilibrium_NS(eps, s, mu)
    pi1_ns = profit1_NS(p1_ns, p2_ns, eps, s, mu)
    pi2_ns = profit2_NS(p1_ns, p2_ns, eps, s, mu)
    regime_ns = classify_regime(p1_ns, p2_ns, eps, s)

    results["Benchmark"] = {
        "p1": p1_ns,
        "p2": p2_ns,
        "pi1": pi1_ns,
        "pi2": pi2_ns,
        "CS": consumer_surplus_NS(p1_ns, p2_ns, eps, s, mu),
        "PS": pi1_ns + pi2_ns,
        "W": total_welfare_NS(p1_ns, p2_ns, eps, s, mu),
        "Theta": 0,
        "regime": regime_ns
    }

    # --- Sharing, rational ---
    p1_r, p2_r, _ = solve_equilibrium(eps, s, gamma=0, mu=mu, sigma=sigma)
    pi1_r = profit1(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma)
    pi2_r = profit2(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma)

    Theta_r = Theta_star(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma)
    regime_r = classify_regime(p1_r, p2_r, eps, s)

    results["Full model (γ=0)"] = {
        "p1": p1_r,
        "p2": p2_r,
        "pi1": pi1_r,
        "pi2": pi2_r,
        "CS": consumer_surplus(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma),
        "PS": pi1_r + pi2_r,
        "W": total_welfare(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma),
        "Theta": Theta_r,
        "regime": regime_r
    }

    # --- Sharing, naive ---
    p1_n, p2_n, _ = solve_equilibrium(eps, s, gamma=0.3, mu=mu, sigma=sigma)
    pi1_n = profit1(p1_n, p2_n, eps, s, gamma=0.3, mu=mu, sigma=sigma)
    pi2_n = profit2(p1_n, p2_n, eps, s, gamma=0.3, mu=mu, sigma=sigma)
    Theta_n = Theta_star(p1_n, p2_n, eps, s, gamma=0.3, mu=mu, sigma=sigma)
    regime_n = classify_regime(p1_n, p2_n, eps, s)

    results["Full model (γ=0.3)"] = {
        "p1": p1_n,
        "p2": p2_n,
        "pi1": pi1_n,
        "pi2": pi2_n,
        "CS": consumer_surplus(p1_n, p2_n, eps, s, gamma=0.3, mu=mu, sigma=sigma),
        "PS": pi1_n + pi2_n,
        "W": total_welfare(p1_n, p2_n, eps, s, gamma=0.3, mu=mu, sigma=sigma),
        "Theta": Theta_n,
        "regime": regime_n
    }

    return pd.DataFrame(results).T

def plot_welfare_comparison(df):
    metrics = ["CS", "PS", "W", "pi1", "pi2"]
    x = np.arange(len(metrics))
    width = 0.25

    colors = ["indigo", "slateblue", "cornflowerblue"]
    models = df.index.tolist()

    fig, ax = plt.subplots(figsize=(8,5))

    for i, model in enumerate(models):
        values = [df.loc[model, m] for m in metrics]
        ax.bar(
            x + i*width,
            values,
            width,
            label=model,
            color=colors[i],
            edgecolor='black',
            linewidth=0.5
        )

    ax.set_xticks(x + width, metrics)
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()
