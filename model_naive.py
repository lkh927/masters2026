# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm

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
    z1_b = 1 + eps - p1 - np.sqrt(2*s)
    z1_w = 1 - p1 - np.sqrt(2*s)
    return z1_b, z1_w

def z2(p2, eps, s):
    z2_w = 1 - p2 - np.sqrt(2*s)
    z2_b = 1 + eps - p2 - np.sqrt(2*s)
    return z2_w, z2_b

# EXPECTED UTILITIES AND DISCLOSURE CUTOFF #
def EU1_B(p1, p2, eps, s):
    z_2, _ = z2(p2, eps, s)
    if z_2 > 0:
        EU1 = z_2*(z_2 - eps + p1) + ((1 + eps - p1)**2 - z_2**2)/2
    else:
         EU1 = (1 + eps - p1)**2/2
    return EU1

def EU1_W(p1, p2, eps, s):
    _, z_2 = z2(p2, eps, s)
    if z_2 > 0:
        EU1 = z_2*(z_2 + p1) + ((1 - p1)**2 - z_2**2)/2
    else:
         EU1 = (1 - p1)**2/2
    return EU1

def EU2_W(p1, p2, eps, s):
    z_1, _ = z1(p1, eps, s)
    if z_1 > 0:
        EU2 = z_1*(z_1 + p2) + ((1-p2)**2 - z_1**2)/2
    else:
        EU2 = (1 - p2)**2/2
    return EU2

def EU2_B(p1, p2, eps, s):
    _, z_1 = z1(p1, eps, s)
    if z_1 > 0:
        EU2 = z_1*(z_1 - eps + p2) + ((1 + eps - p2)**2 - z_1**2)/2
    else:
        EU2 = (1 + eps - p2)**2/2
    return EU2

def theta_soph_1(p1, p2, eps, s, sigma):
    theta_1 = (EU1_B(p1, p2, eps, s) - EU2_W(p1, p2, eps, s))/2
    return np.clip(theta_1/sigma, 0, 1)

def theta_soph_2(p1, p2, eps, s, sigma):
    theta_2 = (EU2_B(p1, p2, eps, s) - EU1_W(p1, p2, eps, s))/2
    return np.clip(theta_2/sigma, 0, 1)

def theta_naive_1(p1, p2, eps, s, sigma, alpha):
    theta_1 = theta_soph_1(p1, p2, eps, s, sigma)
    theta_1_naive = theta_1/alpha
    return np.clip(theta_1_naive, 0, 1)

def theta_naive_2(p1, p2, eps, s, sigma, alpha):
    theta_2 = theta_soph_2(p1, p2, eps, s, sigma)
    theta_2_naive = theta_2/alpha
    return np.clip(theta_2_naive, 0, 1)

def theta_star_1(p1, p2, eps, s, gamma, sigma, alpha):
    sophisticated = theta_soph_1(p1, p2, eps, s, sigma)
    naive = theta_naive_1(p1, p2, eps, s, sigma, alpha)
    theta_star = gamma*naive + (1-gamma)*sophisticated
    return np.clip(theta_star, 0, 1)

def theta_star_2(p1, p2, eps, s, gamma, sigma, alpha):
    sophisticated = theta_soph_2(p1, p2, eps, s, sigma)
    naive = theta_naive_2(p1, p2, eps, s, sigma, alpha)
    theta_star = gamma*naive + (1-gamma)*sophisticated
    return np.clip(theta_star, 0, 1)

def Theta_star(p1, p2, eps, s, gamma, mu, sigma, alpha):
    Theta = mu * theta_star_1(p1, p2, eps, s, gamma, sigma, alpha) + (1-mu) * theta_star_2(p1, p2, eps, s, gamma, sigma, alpha)
    return Theta

# RANKING PROBABILITIES #
def ranking_probs(p1, p2, eps, s, gamma, mu, sigma, alpha):
    theta1 = theta_star_1(p1, p2, eps, s, gamma, sigma, alpha)
    theta2 = theta_star_2(p1, p2, eps, s, gamma, sigma, alpha)
    
    non_disclosure = mu*(1-theta1) + (1-mu)*(1-theta2)
    
    R1_first = mu*theta1 + 0.5 * non_disclosure
    R2_first = (1-mu)*theta2 + 0.5 * non_disclosure

    return R1_first, R2_first

# DEMAND #
# interior case relies on positive reservation values and 0 is in [eps-p_1, 1+eps-p_1] and [-p_2, 1-p_2]

def D1_F(p1, p2, eps, s, gamma, mu, sigma, alpha):
    thetastar_1 = theta_star_1(p1, p2, eps, s, gamma, sigma, alpha)
    thetastar_2 = theta_star_2(p1, p2, eps, s, gamma, sigma, alpha)
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

def D1_R(p1, p2, eps, s, gamma, mu, sigma, alpha):
    thetastar_1 = theta_star_1(p1, p2, eps, s, gamma, sigma, alpha)
    thetastar_2 = theta_star_2(p1, p2, eps, s, gamma, sigma, alpha)
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

def D1(p1, p2, eps, s, gamma, mu, sigma, alpha):
    D1F = D1_F(p1, p2, eps, s, gamma, mu, sigma, alpha)
    D1R = D1_R(p1, p2, eps, s, gamma, mu, sigma, alpha)
    return D1F + D1R

def D2_F(p1, p2, eps, s, gamma, mu, sigma, alpha):
    thetastar_1 = theta_star_1(p1, p2, eps, s, gamma, sigma, alpha)
    thetastar_2 = theta_star_2(p1, p2, eps, s, gamma, sigma, alpha)
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

def D2_R(p1, p2, eps, s, gamma, mu, sigma, alpha):
    thetastar_1 = theta_star_1(p1, p2, eps, s, gamma, sigma, alpha)
    thetastar_2 = theta_star_2(p1, p2, eps, s, gamma, sigma, alpha)
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

def D2(p1, p2, eps, s, gamma, mu, sigma, alpha):
    D2F = D2_F(p1, p2, eps, s, gamma, mu, sigma, alpha)
    D2R = D2_R(p1, p2, eps, s, gamma, mu, sigma, alpha)
    
    return D2F + D2R

# FIRM PROFITS #
def profit1(p1, p2, eps, s, gamma, mu, sigma, alpha):
    return p1 * D1(p1, p2, eps, s, gamma, mu, sigma, alpha)

def profit2(p1, p2, eps, s, gamma, mu, sigma, alpha):
    return p2 * D2(p1, p2, eps, s, gamma, mu, sigma, alpha)

# BEST RESPONSE FUNCTIONS #
def BR1(p2, eps, s, gamma, mu, sigma, alpha, n=200):
    grid = np.linspace(0, 1+eps, n)
    profits = [profit1(p, p2, eps, s, gamma, mu, sigma, alpha) for p in grid]
    p0 = grid[np.argmax(profits)]

    res = minimize_scalar(
        lambda p: -profit1(p, p2, eps, s, gamma, mu, sigma, alpha),
        bounds = (max(0, p0-0.1), min(1+eps, p0+0.1)),
        method='bounded')
    return res.x

def BR2(p1, eps, s, gamma, mu, sigma, alpha, n=200):
    grid = np.linspace(0, 1+eps, n)
    profits = [profit2(p1, p, eps, s, gamma, mu, sigma, alpha) for p in grid]
    p0 = grid[np.argmax(profits)]

    res = minimize_scalar(
        lambda p: -profit2(p1, p, eps, s, gamma, mu, sigma, alpha),
        bounds = (max(0, p0-0.1), min(1+eps, p0+0.1)),
        method='bounded')
    return res.x

# EQUILIBRIUM SOLVER #
def solve_equilibrium(eps, s, gamma, mu, sigma, alpha, p1_init=0.5, p2_init=0.5, tol=1e-6, max_iter=500):
    
    p1, p2 = p1_init, p2_init
    
    for i in range(max_iter):
        p1_new = BR1(p2, eps, s, gamma, mu, sigma, alpha)
        p2_new = BR2(p1_new, eps, s, gamma, mu, sigma, alpha)
        
        if max(abs(p1_new - p1), abs(p2_new - p2)) < tol:
            return p1_new, p2_new, True
        
        p1, p2 = p1_new, p2_new
    
    return p1, p2, False

def check_interior(p1, p2, eps, s, gamma, mu, sigma, alpha):
    z1_A, z1_B = z1(p1, eps, s)
    z2_A, z2_B= z2(p2, eps, s)
    theta = Theta_star(p1, p2, eps, s, gamma, mu, sigma, alpha)
    
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

# WELFARE OUTCOMES #

def consumer_surplus(p1, p2, eps, s, gamma, mu, sigma, alpha):
    
    theta1 = theta_star_1(p1, p2, eps, s, gamma, sigma, alpha)
    theta2 = theta_star_2(p1, p2, eps, s, gamma, sigma, alpha)
    theta1_s = theta_soph_1(p1, p2, eps, s, sigma)
    theta2_s = theta_soph_2(p1, p2, eps, s, sigma)
    theta1_n = theta_naive_1(p1, p2, eps, s, sigma, alpha)
    theta2_n = theta_naive_2(p1, p2, eps, s, sigma, alpha)
    
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
    cost_A = ((1-gamma) * theta1_s**2 / 2 + gamma * theta1_n**2 / 2)

    cost_B = ((1-gamma) * theta2_s**2 / 2 + gamma * theta2_n**2 / 2)
    
    return mu * (EU_A - cost_A) + (1-mu) * (EU_B - cost_B)

def producer_surplus(p1, p2, eps, s, gamma, mu, sigma, alpha):
    return profit1(p1, p2, eps, s, gamma, mu, sigma, alpha) + profit2(p1, p2, eps, s, gamma, mu, sigma, alpha)

def total_welfare(p1, p2, eps, s, gamma, mu, sigma, alpha):
    return consumer_surplus(p1, p2, eps, s, gamma, mu, sigma, alpha) + \
           producer_surplus(p1, p2, eps, s, gamma, mu, sigma, alpha)


##### EQUILIBRIUM CALIBRATION #####

def classify_regime(p1, p2, eps, s):
    z1_b, z1_w = z1(p1, eps, s)
    z2_w, z2_b = z2(p2, eps, s)
    
    # for consumers who prefer firm 1:
    if z2_w > 0 and z1_b > 0:
        regime_1 = "A"      # full search
    elif z2_w <= 0 and z1_b > 0:
        regime_1 = "AN"     # no search after good match
    elif z2_w > 0 and z1_b <= 0:
        regime_1 = "NA"     # no search after bad match
    else:
        regime_1 = "N"      # no search at all

    # for consumers who prefer firm 2:
    if z1_w > 0 and z2_b > 0:
        regime_2 = "A"      # full search
    elif z1_w <= 0 and z2_b > 0:
        regime_2 = "AN"     # no search after good match
    elif z1_w > 0 and z2_b <= 0:
        regime_2 = "NA"     # no search after bad match
    else:
        regime_2 = "N"      # no search at all
    
    # aggregate regime
    if regime_1 == regime_2:
        regime = regime_1
    else:
        regime = f"{regime_1}/{regime_2}"

    return regime


def equilibrium_path_s_eps(eps_grid, s_grid, gamma, mu, sigma, alpha, prev_p1=0.5, prev_p2=0.5):
    results = []

    for eps in eps_grid:
        for s in s_grid:
            try:
                p1, p2, converged = solve_equilibrium(eps, s, gamma, mu, sigma, alpha,
                    p1_init=prev_p1,p2_init=prev_p2)

                # warm start update
                prev_p1, prev_p2 = p1, p2

                # key objects
                Theta = Theta_star(p1, p2, eps, s, gamma, mu, sigma, alpha)
                interior = check_interior(p1, p2, eps, s, gamma, mu, sigma, alpha)["interior"]
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

def plot_regime_map(df, eps_grid, s_grid):
    regime_order = ["A", "A/AN", "A/NA", "A/N", "AN", "AN/A", "AN/NA", "AN/N", "NA", "NA/A", "NA/AN", "NA/N", "N/A", "N/AN", "N/NA", "N"]
    
    # collect all regimes that actually appear
    unique_regimes = [r for r in regime_order if r in df["regime"].unique()]
    # assign integer code dynamically
    regime_dict = {reg: i for i, reg in enumerate(unique_regimes)}

    # map to integers
    Z = df.assign(regime_code=df["regime"].map(regime_dict)
                  ).pivot(index="s", columns="eps", values="regime_code")

    colors = plt.cm.Blues(np.linspace(0.25, 0.95, len(unique_regimes)))
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(unique_regimes) + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8,6))
    mesh = ax.pcolormesh(eps_grid, s_grid, Z.values, shading='nearest',
        cmap=cmap, norm=norm)

    # colorbar
    cbar = plt.colorbar(mesh, ticks=np.arange(len(unique_regimes)), boundaries=bounds, spacing='proportional')
    cbar.set_ticklabels(unique_regimes)

    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"search cost $s$")

    plt.tight_layout()
    plt.show()

def plot_price_curves(eps_grid, s_values, gamma, mu, sigma, alpha, p1_init=0.5, p2_init=0.5):
    fig, axes = plt.subplots(1, 3,figsize=(18,5))

    for i, s in enumerate(s_values):
        ax = axes[i]

        p1_list, p2_list, theta_list = [], [], []
        regimes = []

        p1_curr, p2_curr = p1_init, p2_init
        for eps in eps_grid:

            p1, p2, _ = solve_equilibrium(eps, s, gamma, mu, sigma, alpha, p1_init=p1_curr, p2_init=p2_curr)
            theta = Theta_star(p1, p2, eps, s, gamma, mu, sigma, alpha)
            regime= classify_regime(p1, p2, eps, s)

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

def plot_price_curves_s(s_grid, eps_values, gamma, mu, sigma, alpha, p1_init=0.5, p2_init=0.5):
    fig, axes = plt.subplots(1, len(eps_values), figsize=(18,5), sharey=True)

    for i, eps in enumerate(eps_values):
        ax = axes[i]

        p1_list, p2_list, theta_list = [], [], []
        regimes = []        
        
        # reset warm start for each epsilon
        p1_curr, p2_curr = p1_init, p2_init
        for s in s_grid:
            p1, p2, _ = solve_equilibrium(eps, s, gamma, mu, sigma, alpha,
                p1_init=p1_curr, p2_init=p2_curr)
            theta = Theta_star(p1, p2, eps, s, gamma, mu, sigma, alpha)
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



###### DEMAND COMPOSITION #####
def demand_components_1(p1, p2, eps, s, gamma, mu, sigma, alpha):

    theta1 = theta_star_1(p1, p2, eps, s, gamma, sigma, alpha)
    theta2 = theta_star_2(p1, p2, eps, s, gamma, sigma, alpha)
    z1_b, z1_w = z1(p1, eps, s)
    z2_w, z2_b = z2(p2, eps, s)

    # Immediate purchase fresh demand #
    # Firm 1 is preferred: B=1
    if z2_w > 0:
        imm_A = (1 + theta1)/2 * (1 + eps - p1 - z2_w)
    else:
        imm_A = (1 + theta1)/2 * (1 + eps - p1)

    # Firm 2 is preferred: W=1
    if z2_b > 0:
        imm_B = (1 - theta2)/2 * (1 - z2_b - p1)
    else:
        imm_B = (1 - theta2)/2 * (1 - p1)

    D_immediate = mu * imm_A + (1-mu) * imm_B

    # Search and purchase fresh demand #
    # Firm 1 is preferred: B=1
    if z1_b > 0:
        search_A = (1 - theta1)/2 * ((p2 + z1_b)*(1 + eps - p1) - z1_b**2/2)
    else:
        search_A = 0

    # Firm 2 is preferred: W=1
    if z1_w > 0:
        search_B = (1 + theta2)/2 * ((p2 - eps + z1_w)*(1 - p1) - z1_w**2/2)
    else:
        search_B = 0

    D_search = mu * search_A + (1-mu) * search_B

    # Reurn demand
    D_return = D1_R(p1, p2, eps, s, gamma, mu, sigma, alpha)

    return D_immediate, D_search, D_return

def demand_components_2(p1, p2, eps, s, gamma, mu, sigma, alpha):

    theta1 = theta_star_1(p1, p2, eps, s, gamma, sigma, alpha)
    theta2 = theta_star_2(p1, p2, eps, s, gamma, sigma, alpha)

    z1_b, z1_w = z1(p1, eps, s)
    z2_w, z2_b = z2(p2, eps, s)

    # Immediate purchase fresh demand #
    # Firm 1 is preferred: W=2
    if z1_b > 0:
        imm_A = (1 - theta1)/2 * (1 - z1_b - p2)
    else:
        imm_A = (1 - theta1)/2 * (1 - p2)

    # Firm 2 is preferred: B=2
    if z1_w > 0:
        imm_B = (1 + theta2)/2 * (1 + eps - p2 - z1_w)
    else:
        imm_B = (1 + theta2)/2 * (1 + eps - p2)

    D_immediate = mu * imm_A + (1-mu) * imm_B

   # Search and purchase fresh demand #
    # Firm 1 is preferred: W=2
    if z2_w > 0:
        search_A = (1 + theta1)/2 * ((p1 - eps + z2_w)*(1 - p2) - z2_w**2/2)
    else:
        search_A = 0

    # Firm 2 is preferred: B=2
    if z2_b > 0:
        search_B = (1 - theta2)/2 * ((p1 + z2_b)*(1 + eps - p2) - z2_b**2/2)
    else:
        search_B = 0

    D_search = mu * search_A + (1-mu) * search_B

    # Return demand #
    D_return = D2_R(p1, p2, eps, s, gamma, mu, sigma, alpha)

    return D_immediate, D_search, D_return


def plot_demand_composition_eps(eps_grid, s_values, gamma, mu, sigma, alpha, firm=1, p1_init=0.5, p2_init=0.5):
    fig, axes = plt.subplots(1, 3, figsize=(18,5), sharey=True)

    colors = ["#c7d9f2", "#7ea6e0", "#2f5aa8"]

    for i, s in enumerate(s_values):
        ax = axes[i]

        imm_list, search_list, return_list, regimes = [], [], [], []
       
        p1_curr, p2_curr = p1_init, p2_init
        for eps in eps_grid:
            p1, p2, _ = solve_equilibrium(eps, s, gamma, mu, sigma, alpha,p1_init=p1_curr,p2_init=p2_curr)
            regime = classify_regime(p1, p2, eps, s)
            if firm == 1:
                imm, search, ret = demand_components_1(p1, p2, eps, s, gamma, mu, sigma, alpha)
            else:
                imm, search, ret = demand_components_2(p1, p2, eps, s, gamma, mu, sigma, alpha)

            p1_curr, p2_curr = p1, p2

            imm_list.append(imm)
            search_list.append(search)
            return_list.append(ret)
            regimes.append(regime)

        ax.stackplot(eps_grid,imm_list,search_list,return_list,
            labels=["Immediate purchase","Search then purchase","Return demand"],
            colors=colors,alpha=0.95)

        # regime boundaries
        for j in range(1, len(eps_grid)):
            if regimes[j] != regimes[j-1]:
                boundary = 0.5*(eps_grid[j] + eps_grid[j-1])
                ax.axvline(boundary, color='black',linestyle='--', alpha=0.5)

        ax.set_title(f"Search cost = {s}")
        ax.set_xlabel(r"$\epsilon$")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel("Demand")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5, 1.03),ncol=3,frameon=False)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()


def plot_demand_composition_s(s_grid, eps_values, gamma, mu, sigma, alpha, firm=1, p1_init=0.5,p2_init=0.5):
    fig, axes = plt.subplots(1, 3, figsize=(18,5), sharey=True)

    colors = ["#c7d9f2", "#7ea6e0", "#2f5aa8"]

    for i, eps in enumerate(eps_values):
        ax = axes[i]

        imm_list, search_list, return_list, regimes = [], [], [], []

        p1_curr, p2_curr = p1_init, p2_init
        for s in s_grid:
            p1, p2, _ = solve_equilibrium(eps, s, gamma, mu, sigma, alpha,p1_init=p1_curr,p2_init=p2_curr)
            regime = classify_regime(p1, p2, eps, s)
            if firm == 1:
                imm, search, ret = demand_components_1(p1, p2, eps, s, gamma, mu, sigma, alpha)
            else:
                imm, search, ret = demand_components_2(p1, p2, eps, s, gamma, mu, sigma, alpha)
            
            p1_curr, p2_curr = p1, p2
            
            imm_list.append(imm)
            search_list.append(search)
            return_list.append(ret)
            regimes.append(regime)

        ax.stackplot(s_grid,imm_list,search_list,return_list,
            labels=["Immediate purchase","Search then purchase","Return demand"],
            colors=colors, alpha=0.95)

        for j in range(1, len(s_grid)):
            if regimes[j] != regimes[j-1]:
                boundary = 0.5*(s_grid[j] + s_grid[j-1])
                ax.axvline(boundary, color='black',linestyle='--', alpha=0.5)

        ax.set_title(rf"$\epsilon = {eps}$")
        ax.set_xlabel(r"$s$")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel("Demand")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5, 1.03),ncol=3,frameon=False)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

##### Derivatives of theta^* #####

def dtheta_dp1(p1, p2, eps, s, mu, sigma):
    z1_b, z1_w = z1(p1, eps, s)
    z2_w, z2_b = z2(p2, eps, s)

    latent_theta1 = ((EU1_B(p1, p2, eps, s) - EU2_W(p1, p2, eps, s))/2)/sigma
    latent_theta2 = ((EU2_B(p1, p2, eps, s) - EU1_W(p1, p2, eps, s))/2)/sigma

    # Firm 1 is preferred: B=1
    if z1_b > 0 and z2_w > 0: # reg A (active search)
        dtheta_b1 = 1/(2*sigma) * (1 - 2*np.sqrt(2*s))
    elif z1_b > 0 and z2_w <= 0: # reg AN (no search after good match)
        dtheta_b1 = 1/(2*sigma) * (p2 - np.sqrt(2*s))
    elif z1_b <= 0 and z2_w > 0: # reg NA (no search after bad match)
        dtheta_b1 = 1/(2*sigma) * (p1 - p2 - eps - np.sqrt(2*s))
    elif z1_b <=0 and z2_w <= 0: # reg N (no search)
        dtheta_b1 = 1/(2*sigma) * (p1 - 1 - eps)
    # make sure theta in [0,1]
    if latent_theta1 < 0 or latent_theta1 > 1:
        dtheta_b1 = 0

    # Firm 2 is preferred: w=1
    if z1_w > 0 and z2_b > 0: # reg A (active search)
        dtheta_w1 = 1/(2*sigma) * (2*np.sqrt(2*s) - 1)
    elif z1_w <=0 and z2_b > 0: # reg AN (no search after good match)
        dtheta_w1 = 1/(2*sigma) * (p2 - p1 - eps + np.sqrt(2*s))
    elif z1_w > 0 and z2_b <= 0: # reg NA (no search after bad match)
        dtheta_w1 = 1/(2*sigma) * (np.sqrt(2*s) + eps - p2)
    elif z1_w <= 0 and z2_b <= 0: # reg N (no search)
        dtheta_w1 = 1/(2*sigma) * (1 - p1)
    # make sure theta in [0,1]
    if latent_theta2 < 0 or latent_theta2 > 1:
        dtheta_w1 = 0
    
    dtheta_1 = mu * dtheta_b1 + (1-mu) * dtheta_w1
    return dtheta_1

def dtheta_dp2(p1, p2, eps, s, mu, sigma):
    z1_b, z1_w = z1(p1, eps, s)
    z2_w, z2_b = z2(p2, eps, s)

    latent_theta1 = ((EU1_B(p1, p2, eps, s) - EU2_W(p1, p2, eps, s))/2)/sigma
    latent_theta2 = ((EU2_B(p1, p2, eps, s) - EU1_W(p1, p2, eps, s))/2)/sigma

    # Firm 1 is preferred: W = 2
    if z1_b > 0 and z2_w > 0: # reg A (active search)
        dtheta_w2 = 1/(2*sigma) * (2*np.sqrt(2*s) - 1)
    elif z1_b > 0 and z2_w <= 0: # reg AN (no search after good match)
        dtheta_w2 = 1/(2*sigma) * (p1 - p2 - eps + np.sqrt(2*s))
    elif z1_b <= 0 and z2_w > 0: # reg NA (no search after bad match)
        dtheta_w2 = 1/(2*sigma) * (np.sqrt(2*s) + eps - p1)
    elif z1_b <=0 and z2_w <= 0: # reg N (no search)
        dtheta_w2 = 1/(2*sigma) * (1 - p2)
    # make sure theta in [0,1]
    if latent_theta1 < 0 or latent_theta1 > 1:
        dtheta_w2 = 0 
    
    # Firm 2 is preferred: B = 2
    if z1_w > 0 and z2_b > 0: # reg A (active search)
        dtheta_b2 = 1/(2*sigma) * (1 - 2*np.sqrt(2*s))
    elif z1_w <=0 and z2_b > 0: # reg AN (no search after good match)
        dtheta_b2 = 1/(2*sigma) * (p1 - np.sqrt(2*s))
    elif z1_w > 0 and z2_b <= 0: # reg NA (no search after bad match)
        dtheta_b2 = 1/(2*sigma) * (p2 - p1 - eps - np.sqrt(2*s))
    elif z1_w <= 0 and z2_b <= 0: # reg N (no search)
        dtheta_b2 = 1/(2*sigma) * (p2 - 1 - eps)
    # make sure theta in [0,1]
    if latent_theta2 < 0 or latent_theta2 > 1:
        dtheta_b2 = 0
    
    dtheta_2 = mu * dtheta_w2 + (1-mu) * dtheta_b2
    return dtheta_2

def plot_dtheta(eps_grid, s_values, gamma, mu, sigma, alpha, p1_init=0.5, p2_init=0.5):
    fig, axes = plt.subplots(1, 3,figsize=(18,5))

    for i, s in enumerate(s_values):
        ax = axes[i]
        p1_curr, p2_curr = p1_init, p2_init

        dtheta1_list, dtheta2_list = [], []
        regimes = []
        for eps in eps_grid:
            p1, p2, _ = solve_equilibrium(eps, s, gamma, mu, alpha, sigma,p1_init=p1_curr,p2_init=p2_curr)

            regime = classify_regime(p1, p2, eps, s)
            dtheta1 = dtheta_dp1(p1, p2, eps, s, mu, sigma)
            dtheta2 = dtheta_dp2(p1, p2, eps, s, mu, sigma)

            dtheta1_list.append(dtheta1)
            dtheta2_list.append(dtheta2)
            regimes.append(regime)

            p1_curr, p2_curr = p1, p2

        ax.plot(eps_grid, dtheta1_list, label=r"$\partial \Theta^*/ \partial p_1$", color="indigo", linewidth=2)
        ax.plot(eps_grid, dtheta2_list, label=r"$\partial \Theta^*/ \partial p_2$", color="slateblue", linewidth=2)

        for j in range(1, len(eps_grid)):
            if regimes[j] != regimes[j-1]:
                boundary = 0.5*(eps_grid[j] + eps_grid[j-1])
                ax.axvline(boundary, color='black',linestyle='--', alpha=0.5)

        ax.set_title(rf"Search cost = {s}")
        ax.set_xlabel(r"$\epsilon$")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)

    # combined legend
    lines1, labels1 = axes[0].get_legend_handles_labels()
    fig.legend(lines1, labels1, loc='upper center', ncol=2, frameon=False)
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()

def plot_dtheta_s(s_grid, eps_values, gamma, mu, sigma, alpha, p1_init=0.5, p2_init=0.5):
    fig, axes = plt.subplots(1, 3,figsize=(18,5))

    for i, eps in enumerate(eps_values):
        ax = axes[i]
        p1_curr, p2_curr = p1_init, p2_init

        dtheta1_list, dtheta2_list = [], []
        regimes = []
        for s in s_grid:
            p1, p2, _ = solve_equilibrium(eps, s, gamma, mu, sigma, alpha, p1_init=p1_curr,p2_init=p2_curr)
            regime = classify_regime(p1, p2, eps, s)
            dtheta1 = dtheta_dp1(p1, p2, eps, s, mu, sigma)
            dtheta2 = dtheta_dp2(p1, p2, eps, s, mu, sigma)

            dtheta1_list.append(dtheta1)
            dtheta2_list.append(dtheta2)
            regimes.append(regime)

            p1_curr, p2_curr = p1, p2

        ax.plot(s_grid, dtheta1_list, label=r"$\partial \Theta^*/ \partial p_1$", color="indigo", linewidth=2)
        ax.plot(s_grid, dtheta2_list, label=r"$\partial \Theta^*/ \partial p_2$", color="slateblue", linewidth=2)

        for j in range(1, len(s_grid)):
            if regimes[j] != regimes[j-1]:
                boundary = 0.5*(s_grid[j] + s_grid[j-1])
                ax.axvline(boundary, color='black',linestyle='--', alpha=0.5)

        ax.set_title(rf"$\epsilon$= {eps}")
        ax.set_xlabel(r"$s$")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)

    # combined legend
    lines1, labels1 = axes[0].get_legend_handles_labels()
    fig.legend(lines1, labels1, loc='upper center', ncol=2, frameon=False)
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()

##### NAIVETE #######

def plot_prices_gamma(cases, gamma_grid, mu, sigma, alpha, p1_init=0.5, p2_init=0.5):
    fig, axes = plt.subplots(1,2,figsize=(10,4),sharex=True)

    for i, case in enumerate(cases):
        ax = axes[i]
        eps, s = case
        p1_curr, p2_curr = p1_init, p2_init
        p1_list, p2_list = [], []
        regimes = []

        for gamma in gamma_grid:
            p1, p2, _ = solve_equilibrium(eps, s, gamma, mu, sigma, alpha, p1_init=p1_curr,p2_init=p2_curr)
            # warm start update
            p1_curr, p2_curr = p1, p2
            regime = classify_regime(p1, p2, eps, s)

            p1_list.append(p1)
            p2_list.append(p2)
            regimes.append(regime)

        # regime boundaries
        for j in range(1, len(gamma_grid)):
            if regimes[j] != regimes[j-1]:
                boundary = 0.5 * (gamma_grid[j] + gamma_grid[j-1])
                ax.axvline(boundary, color='black',linestyle='--', alpha=0.5)

        ax.plot(gamma_grid, p1_list, label=r"$p_1$", color="slateblue", linewidth=2)
        ax.plot(gamma_grid, p2_list, label=r"$p_2$", color="royalblue", linewidth=2)
        ax.set_title(rf"$\epsilon$={eps}, $s$={s}")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("Welfare")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0.2,1)

    lines1, labels1 = axes[0].get_legend_handles_labels()
    fig.legend(lines1, labels1, loc='upper center', ncol=2, frameon=False)
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()

def plot_welfare_gamma(cases, gamma_grid, mu, sigma, alpha, p1_init=0.5, p2_init=0.5):
    fig, axes = plt.subplots(1,2,figsize=(10,4),sharex=True)

    for i, case in enumerate(cases):
        ax = axes[i]
        eps, s = case
        p1_curr, p2_curr = p1_init, p2_init
        CS_list, PS_list, W_list, theta_list = [], [], [], []
        regimes = []

        for gamma in gamma_grid:
            p1, p2, _ = solve_equilibrium(eps, s, gamma, mu, sigma, alpha, p1_init=p1_curr,p2_init=p2_curr)
            # warm start update
            p1_curr, p2_curr = p1, p2

            Theta = Theta_star(p1, p2, eps, s, gamma, mu, sigma, alpha)
            regime = classify_regime(p1, p2, eps, s)
            CS = consumer_surplus(p1, p2, eps, s, gamma, mu, sigma, alpha)
            PS = producer_surplus(p1, p2, eps, s, gamma, mu, sigma, alpha)
            W = CS + PS

            CS_list.append(CS)
            PS_list.append(PS)
            W_list.append(W)
            regimes.append(regime)
            theta_list.append(Theta)

        # regime boundaries
        for j in range(1, len(gamma_grid)):
            if regimes[j] != regimes[j-1]:
                boundary = 0.5 * (gamma_grid[j] + gamma_grid[j-1])
                ax.axvline(boundary, color='black',linestyle='--', alpha=0.5)

        ax.plot(gamma_grid, CS_list, label="CS", color="slateblue", linewidth=2)
        ax.plot(gamma_grid, PS_list, label="PS", color="royalblue", linewidth=2)
        ax.plot(gamma_grid, W_list, label="W", color="indigo", linewidth=2)
        ax.plot(gamma_grid, theta_list, label=r"$\Theta^*$", color="cornflowerblue", linestyle='--', linewidth=1)
        ax.set_title(rf"$\epsilon$={eps}, $s$={s}")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("Welfare")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0,1.2)

    lines1, labels1 = axes[0].get_legend_handles_labels()
    fig.legend(lines1, labels1, loc='upper center', ncol=4, frameon=False)
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()


def plot_profit_gamma(cases, gamma_grid, mu, sigma, alpha, p1_init=0.5, p2_init=0.5):
    fig, axes = plt.subplots(1,2,figsize=(10,4),sharex=True)

    for i, case in enumerate(cases):
        ax = axes[i]
        eps, s = case
        p1_curr, p2_curr = p1_init, p2_init
        pi1_list, pi2_list, theta_list = [], [], []
        regimes = []

        for gamma in gamma_grid:
            p1, p2, _ = solve_equilibrium(eps, s, gamma, mu, sigma, alpha, p1_init=p1_curr,p2_init=p2_curr)
            # warm start update
            p1_curr, p2_curr = p1, p2
            Theta = Theta_star(p1, p2, eps, s, gamma, mu, sigma, alpha)
            regime = classify_regime(p1, p2, eps, s)
            pi1 = profit1(p1, p2, eps, s, gamma, mu, sigma, alpha)
            pi2 = profit2(p1, p2, eps, s, gamma, mu, sigma, alpha)

            pi1_list.append(pi1)
            pi2_list.append(pi2)
            regimes.append(regime)

        # regime boundaries
        for j in range(1, len(gamma_grid)):
            if regimes[j] != regimes[j-1]:
                boundary = 0.5 * (gamma_grid[j] + gamma_grid[j-1])
                ax.axvline(boundary, color='black',linestyle='--', alpha=0.5)

        ax.plot(gamma_grid, pi1_list, label=r"$\pi_1$", color="slateblue", linewidth=2)
        ax.plot(gamma_grid, pi2_list, label=r"$\pi_2$", color="royalblue", linewidth=2)
        ax.set_title(rf"$\epsilon$={eps}, $s$={s}")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("Welfare")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0,0.5)

    lines1, labels1 = axes[0].get_legend_handles_labels()
    fig.legend(lines1, labels1, loc='upper center', ncol=4, frameon=False)
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()

###### NO SHARING VERSION OF THE MODEL #######
def D1_F_NS(p1, p2, eps, s, mu):
    z1_b, z1_w = z1(p1, eps, s)
    z2_w, z2_b = z2(p2, eps, s)

    # Firm 1 is preferred: D1_F = DBF_1 (preferred firm, B=1)
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

    # Firm 2 is preferred: D1_F = DWF_1 (non-preferred firm, W=1)
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
    p1_r, p2_r, _ = solve_equilibrium(eps, s, gamma=0, mu=mu, sigma=sigma, alpha=0.2)
    pi1_r = profit1(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma, alpha=0.2)
    pi2_r = profit2(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma, alpha=0.2)

    Theta_r = Theta_star(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma, alpha=0.2)
    regime_r = classify_regime(p1_r, p2_r, eps, s)

    results["Full Model with Disclosure"] = {
        "p1": p1_r,
        "p2": p2_r,
        "pi1": pi1_r,
        "pi2": pi2_r,
        "CS": consumer_surplus(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma, alpha=0.2),
        "PS": pi1_r + pi2_r,
        "W": total_welfare(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma, alpha=0.2),
        "Theta": Theta_r,
        "regime": regime_r
    }
    return pd.DataFrame(results).T

def compare_outcomes(eps, s, mu, sigma, alpha):

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
    p1_r, p2_r, _ = solve_equilibrium(eps, s, gamma=0, mu=mu, sigma=sigma, alpha=alpha)
    pi1_r = profit1(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma, alpha=alpha)
    pi2_r = profit2(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma, alpha=alpha)

    Theta_r = Theta_star(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma, alpha=alpha)
    regime_r = classify_regime(p1_r, p2_r, eps, s)

    results["Full model (γ=0)"] = {
        "p1": p1_r,
        "p2": p2_r,
        "pi1": pi1_r,
        "pi2": pi2_r,
        "CS": consumer_surplus(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma, alpha=alpha),
        "PS": pi1_r + pi2_r,
        "W": total_welfare(p1_r, p2_r, eps, s, gamma=0, mu=mu, sigma=sigma, alpha=alpha),
        "Theta": Theta_r,
        "regime": regime_r
    }

    # --- Sharing, naive ---
    p1_n, p2_n, _ = solve_equilibrium(eps, s, gamma=0.4, mu=mu, sigma=sigma, alpha=alpha)
    pi1_n = profit1(p1_n, p2_n, eps, s, gamma=0.4, mu=mu, sigma=sigma, alpha=alpha)
    pi2_n = profit2(p1_n, p2_n, eps, s, gamma=0.4, mu=mu, sigma=sigma, alpha=alpha)
    Theta_n = Theta_star(p1_n, p2_n, eps, s, gamma=0.4, mu=mu, sigma=sigma, alpha=alpha)
    regime_n = classify_regime(p1_n, p2_n, eps, s)

    results["Full model (γ=0.4)"] = {
        "p1": p1_n,
        "p2": p2_n,
        "pi1": pi1_n,
        "pi2": pi2_n,
        "CS": consumer_surplus(p1_n, p2_n, eps, s, gamma=0.4, mu=mu, sigma=sigma, alpha=alpha),
        "PS": pi1_n + pi2_n,
        "W": total_welfare(p1_n, p2_n, eps, s, gamma=0.4, mu=mu, sigma=sigma, alpha=alpha),
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
        ax.bar(x + i*width, values, width, label=model, color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x + width, metrics)
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Value")
    ax.set_ylim(0, df["W"].values.max()*1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()
