# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize_scalar
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm

def reservation_values(p, eps, s):

    root = np.sqrt(2*s)

    z_b = 1 + eps - p - root
    z_w = 1 - p - root

    return z_b, z_w

def classify_regime(p, eps, s, tol=1e-10):
    z_b, z_w = reservation_values(p, eps, s)
    if z_b > tol and z_w > tol:
        return "A"
    elif z_b > tol and z_w <= tol:
        return "AN"
    elif z_b <= tol and z_w > tol:
        return "NA"
    else:
        return "N"
    
def Theta_symmetric(p, eps, s, sigma):
    z_b, z_w = reservation_values(p, eps, s)
    # Regime A
    if z_b > 0 and z_w > 0:
        theta = (eps*np.sqrt(2*s) - 0.5*eps)
    # Regime AN
    elif z_b > 0 and z_w <= 0:
        theta = (0.25*p**2 + 0.5*np.sqrt(2*s)*(1+eps)- 0.25 - 0.5*eps*p- 0.5*s)
    # Regime NA
    elif z_b <= 0 and z_w > 0:
        theta = (0.25+ 0.5*np.sqrt(2*s)*(eps-1)+ 0.5*s+ 0.25*eps**2- 0.25*p**2)
    # Regime N
    else:
        theta = (0.5*eps*(1-p)+ 0.25*eps**2)

    return np.clip(theta / sigma, 0, 1)

def EU_num(eps, s, p):
    z_b, z_w = reservation_values(p, eps, s)
    if z_w > 0:
        EU_B = z_w*(z_w - eps + p) + ((1 + eps - p)**2 - z_w**2)/2
    else:
        EU_B = (1 + eps - p)**2/2
    
    if z_b > 0:
        EU_W = z_b * (z_b + p) + ((1 - p)**2 - z_b**2)/2
    else:
        EU_W = (1-p)**2/2
    return EU_B, EU_W     

def Theta_symmetric_num(p, eps, s, sigma):
    EU_B, EU_W = EU_num(eps, s, p)
    theta_star = 1/2*(EU_B - EU_W)
    return np.clip(theta_star / sigma, 0, 1)

def ranking_probs(theta):
    r_pref = 0.5*(1 + theta)
    r_nonpref = 0.5*(1 - theta)

    return r_pref, r_nonpref

def demand_symmetric(p, eps, s, sigma):
    #theta = Theta_symmetric(p, eps, s, sigma)
    theta = Theta_symmetric_num(p, eps, s, sigma)
    r_pref, r_nonpref = ranking_probs(theta)
    z_b, z_w = reservation_values(p, eps, s)
    regime = classify_regime(p, eps, s)

    # --------------------------------
    # Preferred consumers
    # --------------------------------

    if regime == "A":

        D_pref = (
            r_pref*(1 + eps - p - z_w)
            + r_nonpref*((p + z_b)*(1 + eps - p) - z_b**2/2)
            + r_pref*(2*z_w*p + z_w**2/2)
        )

    elif regime == "AN":

        D_pref = (
            r_pref*(1 + eps - p)
            + r_nonpref*((p + z_b)*(1 + eps - p) - z_b**2/2)
        )

    elif regime == "NA":

        D_pref = (
            r_pref*(1 + eps - p - z_w)
            + r_pref*(2*z_w*p + z_w**2/2)
        )

    else:

        D_pref = r_pref*(1 + eps - p)

    # --------------------------------
    # Non-preferred consumers
    # --------------------------------

    if regime == "A":

        D_nonpref = (
            r_nonpref*(1 - z_b - p)
            + r_pref*((p - eps + z_w)*(1-p) - z_w**2/2)
            + r_nonpref*(2*z_b*(p-eps) + z_b**2/2)
        )

    elif regime == "AN":

        D_nonpref = (
            r_nonpref*(1 - z_b - p)
            + r_nonpref*(2*z_b*(p-eps) + z_b**2/2)
        )

    elif regime == "NA":

        D_nonpref = (
            r_nonpref*(1-p)
            + r_pref*((p - eps + z_w)*(1-p) - z_w**2/2)
        )

    else:
        D_nonpref = r_nonpref*(1-p)

    return 0.5*(D_pref + D_nonpref)

def demand_symmetric_NS(p, eps, s):
    r_pref, r_nonpref = 0.5, 0.5
    z_b, z_w = reservation_values(p, eps, s)
    regime = classify_regime(p, eps, s)

    # --------------------------------
    # Preferred consumers
    # --------------------------------

    if regime == "A":

        D_pref = (
            r_pref*(1 + eps - p - z_w)
            + r_nonpref*((p + z_b)*(1 + eps - p) - z_b**2/2)
            + r_pref*(2*z_w*p + z_w**2/2)
        )

    elif regime == "AN":

        D_pref = (
            r_pref*(1 + eps - p)
            + r_nonpref*((p + z_b)*(1 + eps - p) - z_b**2/2)
        )

    elif regime == "NA":

        D_pref = (
            r_pref*(1 + eps - p - z_w)
            + r_pref*(2*z_w*p + z_w**2/2)
        )

    else:

        D_pref = r_pref*(1 + eps - p)

    # --------------------------------
    # Non-preferred consumers
    # --------------------------------

    if regime == "A":

        D_nonpref = (
            r_nonpref*(1 - z_b - p)
            + r_pref*((p - eps + z_w)*(1-p) - z_w**2/2)
            + r_nonpref*(2*z_b*(p-eps) + z_b**2/2)
        )

    elif regime == "AN":

        D_nonpref = (
            r_nonpref*(1 - z_b - p)
            + r_nonpref*(2*z_b*(p-eps) + z_b**2/2)
        )

    elif regime == "NA":

        D_nonpref = (
            r_nonpref*(1-p)
            + r_pref*((p - eps + z_w)*(1-p) - z_w**2/2)
        )

    else:
        D_nonpref = r_nonpref*(1-p)

    return 0.5*(D_pref + D_nonpref)

def profit_symmetric(p, eps, s, sigma):
    return p * demand_symmetric(p, eps, s, sigma)

def profit_symmetric_NS(p, eps, s):
    return p * demand_symmetric_NS(p, eps, s)

def solve_symmetric_equilibrium(eps, s, sigma, n=200):
    grid = np.linspace(0, 1+eps, n)
    profits = [profit_symmetric(p, eps, s, sigma) for p in grid]
    p0 = grid[np.argmax(profits)]
    
    obj = lambda p: -profit_symmetric(p, eps, s, sigma)

    res = minimize_scalar(
        obj,
        bounds=(max(0, p0-0.2), min(1+eps, p0+0.1)),
        method='bounded'
    )
    return res.x, res.success

def solve_equilibrium_NS_sym(eps, s, n=200):
    grid = np.linspace(0, 1+eps, n)
    profits = [profit_symmetric_NS(p, eps, s) for p in grid]
    p0 = grid[np.argmax(profits)]
    
    obj = lambda p: -profit_symmetric_NS(p, eps, s)

    res = minimize_scalar(
        obj,
        bounds=(max(0, p0-0.2), min(1+eps, p0+0.1)),
        method='bounded'
    )
    return res.x, res.success

def equilibrium_path_s_eps(eps_grid, s_grid, sigma):
    results = []

    for eps in eps_grid:
        for s in s_grid:
            try:
                p, converged = solve_symmetric_equilibrium(eps, s, sigma)
                #theta = Theta_symmetric(p, eps, s, sigma)
                theta = Theta_symmetric_num(p, eps, s, sigma)
                regime = classify_regime(p, eps, s)

                results.append({
                    "eps": eps,
                    "s": s,
                    "p": p,
                    "Theta": theta,
                    "regime": regime,
                    "converged": converged
                })

            except Exception:
                results.append({
                    "eps": eps,
                    "s": s,
                    "p": np.nan,
                    "Theta": np.nan,
                    "regime": "fail",
                    "converged": False
                })

    return pd.DataFrame(results)

def plot_colorblock(df, eps_grid, s_grid):

    p = df.pivot(index='s', columns='eps', values='p')
    theta = df.pivot(index='s', columns='eps', values='Theta')

    fig, axes = plt.subplots(1, 2, figsize=(11,4.5))

    plots = [
        (p, r"Symmetric price $p^*$"),
        (theta, r"Disclosure share $\Theta^*$")
    ]

    for ax, (data, title) in zip(axes, plots):

        cont = ax.contourf(
            eps_grid,
            s_grid,
            data.values,
            levels=20,
            cmap=plt.cm.Blues
        )

        ax.set_title(title)
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel(r"$s$")

        fig.colorbar(cont, ax=ax)

    plt.tight_layout()
    plt.show()

def plot_regime_map(df, eps_grid, s_grid):
    regime_order = ["A", "AN", "NA", "N"]
    unique_regimes = [r for r in regime_order if r in df["regime"].unique()]
    regime_dict = {reg: i for i, reg in enumerate(unique_regimes)}

    Z = (df.assign(regime_code=df["regime"].map(regime_dict)
            ).pivot(index="s", columns="eps", values="regime_code"))

    colors = plt.cm.Blues(np.linspace(0.25, 0.95, len(unique_regimes)))
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(unique_regimes) + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8,6))

    mesh = ax.pcolormesh(eps_grid, s_grid,Z.values,
        shading='nearest',cmap=cmap, norm=norm)

    cbar = plt.colorbar(mesh, ticks=np.arange(len(unique_regimes)),
            boundaries=bounds, spacing='proportional')
    cbar.set_ticklabels(unique_regimes)

    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$s$")

    plt.tight_layout()
    plt.show()

def consumer_surplus(p, eps, s, sigma):
    theta = Theta_symmetric(p, eps, s, sigma)
    r_pref, r_nonpref = ranking_probs(theta)

    EU_B, EU_W = EU_num(eps, s, p)
    EU = r_pref*EU_B + r_nonpref*EU_W

    cost = theta**2/2

    return EU - cost

def consumer_surplus_NS(p, eps, s):
    EU_B, EU_W = EU_num(eps, s, p)
    EU = 0.5*EU_B + 0.5*EU_W
    return EU

def compare_outcomes_benchmark(eps, s, sigma):

    results = {}
    # --- No sharing ---
    p_ns, _ = solve_equilibrium_NS_sym(eps, s)
    pi_ns = profit_symmetric_NS(p_ns, eps, s)
    cs_ns = consumer_surplus_NS(p_ns, eps, s)
    regime_ns = classify_regime(p_ns, eps, s)

    results["Benchmark: No Disclosure"] = {
        "p1": p_ns,
        "p2": p_ns,
        "pi1": pi_ns,
        "pi2": pi_ns,
        "CS": cs_ns,
        "PS": pi_ns*2,
        "W": cs_ns + (pi_ns*2),
        "Theta": 0,
        "regime": regime_ns
    }

    # --- Sharing, rational ---
    p_r, _ = solve_symmetric_equilibrium(eps, s, sigma)
    pi_r = profit_symmetric(p_r, eps, s, sigma)
    cs_r = consumer_surplus(p_r, eps, s, sigma)

    Theta_r = Theta_symmetric(p_r, eps, s, sigma)
    regime_r = classify_regime(p_r, eps, s)

    results["Full Model with Disclosure"] = {
        "p1": p_r,
        "p2": p_r,
        "pi1": pi_r,
        "pi2": pi_r,
        "CS": cs_r,
        "PS": pi_r*2,
        "W": cs_r+(pi_r*2),
        "Theta": Theta_r,
        "regime": regime_r
    }
    return pd.DataFrame(results).T

def dtheta_p(p, eps, s, sigma):
    regime = classify_regime(p, eps, s)
    if regime == "A":
        dtheta = 0
    elif regime == "AN":
        dtheta = 1/(2*sigma)*(p - eps)
    elif regime == "NA":
        dtheta = -1/(2*sigma)*p
    else:
        dtheta = -1/(2*sigma)*eps

    return dtheta

def plot_dtheta(eps_grid, s_values, sigma):
    fig, axes = plt.subplots(1, 3,figsize=(18,5))

    for i, s in enumerate(s_values):
        ax = axes[i]
        dtheta_list = []
        regimes = []

        for eps in eps_grid:
            p, _ = solve_symmetric_equilibrium(eps, s, sigma)
            dtheta = dtheta_p(p, eps, s, sigma)
            regime = classify_regime(p, eps, s)

            dtheta_list.append(dtheta)
            regimes.append(regime)
        
        ax.plot(eps_grid, dtheta_list, label=r"$\partial\Theta^*/ \partial p$", color="indigo", linewidth = 2)
    
        for j in range(1, len(eps_grid)):
            if regimes[j] != regimes[j-1]:
                boundary = 0.5*(eps_grid[j] + eps_grid[j-1])
                ax.axvline(boundary, color='black',linestyle='--', alpha=0.5)
        
        ax.set_title(f"Search cost = {s}")
        ax.set_xlabel(r"$\epsilon$")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
    
    lines1, labels1 = axes[0].get_legend_handles_labels()
    fig.legend(lines1, labels1, loc='upper center', ncol=2, frameon=False)
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()

def plot_price_curves(eps_grid, s_values, sigma):
    fig, axes = plt.subplots(1, 3,figsize=(18,5))

    for i, s in enumerate(s_values):
        ax = axes[i]

        p_list, theta_list = [], []
        regimes = []

        for eps in eps_grid:

            p, _ = solve_symmetric_equilibrium(eps, s, sigma)
            theta = Theta_symmetric_num(p, eps, s, sigma)
            regime= classify_regime(p, eps, s)

            p_list.append(p)
            theta_list.append(theta)
            regimes.append(regime)

        # --- prices ---
        ax.plot(eps_grid, p_list, label=r"$p$", color="indigo", linewidth=2)

        # regime boundaries
        for j in range(1, len(eps_grid)):
            if regimes[j] != regimes[j-1]:
                boundary = 0.5 * (eps_grid[j] + eps_grid[j-1])
                ax.axvline(boundary, color='black',linestyle='--', alpha=0.5)

        ax.set_title(rf"Search cost = {s}")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylim(0.3, 0.8)
        ax.grid(axis='y',linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel("Price")
        ax.tick_params(axis='y', labelleft=True)

        # aligned ticks
        price_ticks = np.arange(0.2, 0.81, 0.2)
        ax.set_yticks(price_ticks)

        # --- disclosure axis ---
        ax2 = ax.twinx()
        ax2.plot(eps_grid, theta_list, color="cornflowerblue", linestyle='--', linewidth=2.5, label=r"$\Theta^*$")
        ax2.set_ylim(0, 0.6)
        ax2.set_yticks(price_ticks - 0.2)
        ax2.spines['top'].set_visible(False)

        ax2.set_ylabel(r"Disclosure share $\Theta^*$")

    # combined legend
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    fig.legend(lines1 + lines2,labels1 + labels2,loc='upper center',ncol=3,frameon=False)
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()
    
###### DEMAND COMPOSITION #####
def demand_component(p, eps, s, sigma):

    theta = Theta_symmetric_num(p, eps, s, sigma)
    z_b, z_w = reservation_values(p, eps, s)

    # Immediate purchase fresh demand #
    if z_w > 0:
        imm_A = (1 + theta)/2 * (1 + eps - p - z_w)
    else:
        imm_A = (1 + theta)/2 * (1 + eps - p)

    if z_b > 0:
        imm_B = (1 - theta)/2 * (1 - z_b - p)
    else:
        imm_B = (1 - theta)/2 * (1 - p)

    D_immediate = 1/2 * (imm_A + imm_B)

    # Search and purchase fresh demand #
    if z_b > 0:
        search_A = (1 - theta)/2 * ((p + z_b)*(1 + eps - p) - z_b**2/2)
    else:
        search_A = 0

    if z_w > 0:
        search_B = (1 + theta)/2 * ((p - eps + z_w)*(1 - p) - z_w**2/2)
    else:
        search_B = 0

    D_search = 1/2 * (search_A + search_B)

    # Reurn demand
    if z_w > 0:
        return_A = (1+theta)/2 * (2*z_w*p + z_w**2/2)
    else:
        return_A = 0
    if z_b > 0:
        return_B = (1-theta)/2 * (2*z_b*(p - eps) + z_b**2/2)
    else:
        return_B = 0
    D_return = 1/2 * (return_A + return_B)

    return D_immediate, D_search, D_return

def plot_demand_composition_eps(eps_grid, s_values, sigma):
    fig, axes = plt.subplots(1, 3, figsize=(18,5), sharey=True)

    colors = ["#c7d9f2", "#7ea6e0", "#2f5aa8"]

    for i, s in enumerate(s_values):
        ax = axes[i]

        imm_list, search_list, return_list, regimes = [], [], [], []
        for eps in eps_grid:
            p, _ = solve_symmetric_equilibrium(eps, s, sigma)
            regime = classify_regime(p, eps, s)
            imm, search, ret = demand_component(p, eps, s, sigma)

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
