# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from scipy.optimize import curve_fit

# %%
import acnt

# %% [markdown]
# # 1. aCNT fits of HS and four different phase transitions

# %% [markdown]
# ## 1a. HS

# %%
from scipy.special import logsumexp
acnt_fits = True
cnt_fits = True
plotx = 'n' #False
portrait = True
acnt.dmu = dmu = 0.54
acnt.rho_s = rho_s = 1.136 
xlims = np.array([80, 120, 160])
xlims = np.array([100, 150, 200])
if plotx == 'r':
    xlims = (xlims/acnt.rho_s / (4*np.pi/3))**(1/3)
    width_ratios = [1,1,1]
else:
    width_ratios = [100, 150, 200]
params_acnt = []
params_gcnt = []

nplots = 3

if nplots == 2:
    xlims = np.array([150, 150, 150])
    width_ratios = [1,1,1]

if portrait:
    fig, axs = plt.subplots(
        nplots, 1, sharex=True,
        figsize=(3.375,2*3.375))
else:
    fig, axs = plt.subplots(
        1,nplots, sharey=True,
        gridspec_kw={'width_ratios': width_ratios[:nplots]}, figsize=(2*3.375,3))

N_ = np.linspace(0, 200, num=1000)
shift = 0
alphas = np.linspace(-0.5, 0.5)
gammacs = [np.zeros_like(alphas), np.zeros_like(alphas), np.zeros_like(alphas), np.zeros_like(alphas)]

for i, xi in enumerate([8,7,6]):
    # plot umbrella sampling barrier
    df = pd.read_csv(f'../../results/lit/filion_2010_xi{xi}.csv', decimal='.', delimiter=',', names=['N', 'G'])

    order = np.argsort(df['N'])
    N = df['N'][order]
    G = df['G'][order]
    Nc = np.exp(logsumexp(G, b=N) - logsumexp(G))
    Gc = np.max(G)
    print(Gc)
    label = 'US' if i == 2 else None
    ax = axs[i] if len(axs) == 3 else axs[1]
    if plotx == 'r':
        ax.plot((N/rho_s / (4*np.pi/3))**(1/3), G+shift*xi, '-', label=label, c='k')
    elif plotx == 'n':
        ax.plot(N, G+shift*xi, '-', label=label, c='k')
    sigma = 0.1 + 10.0 * (N > N[np.argmax(G)])

    # fit barrier, assuming CNT and only critical nucleus size
    print(Nc, Gc)
    N_fit = [0, Nc]
    G_fit = [0, 0.5*Nc*dmu]
    params, cov = curve_fit(acnt.cnt, N_fit, G_fit, p0=[1,],)
    label = 'CNT' if i == 2 else None
    ax = axs[i] if len(axs) == 3 else axs[0]
    if plotx == 'r':
        ax.plot((N_/rho_s / (4*np.pi/3))**(1/3), acnt.cnt(N_, *params), '--', label=None, c='C3')
    else:
        ax.plot(N_, acnt.cnt(N_, *params), ':', lw=2, label=None, c='C3')
    ax.plot(Nc, 0.5*Nc*dmu, c='C3', label=label, ls=':', lw=2, marker='o')


    # fit barrier with a-CNT
    mask = (N < Nc) * (G > 10.0)
    N_fit, G_fit = N[mask], G[mask]
    params, cov = curve_fit(acnt.r_pol_g0, N_fit, G_fit, p0=[1, 0, 0])
    print('acnt', params)
    params_acnt.append(params)
    G2 = acnt.r_pol_g0(N_, *params)
    label = 'aCNT' if i == 2 else None
    ax = axs[i] if len(axs) == 3 else axs[1]
    if plotx == 'r':
        ax.plot((N_/rho_s / (4*np.pi/3))**(1/3), G2+shift*xi, '-', label=label, c='C0')
    elif plotx == 'n':
        ax.plot(N_, G2+shift*xi, '-', label=label, c='C0', ls='--')
    if i == 1:
        params_filion = params

    axs[i].spines[['right', 'top']].set_visible(False)
    
       
    ax = axs[min(i,nplots-1)]
    if plotx == 'n':
        ax.set_xlabel(rf'$n ~(\xi_c = {xi})$')
        ax.set_xticks(np.arange(0, xlims[i]+50, 50), labels=np.arange(0, xlims[i]+50, 50))
        ax.set_xlim(0, xlims[i])
    if plotx == 'r':
        ax.set_xlim(0, 3.5)
    if plotx:
        ax.set_ylim(0, 30)


if portrait:
    axs[2].legend(ncol=1)
else:
    fig.legend(ncol=1, bbox_to_anchor=(0, 0.08, 0.96, 0.4))
for ax in axs:
    ax.set_ylabel(r'$\Delta G ~/~ k_B T$')
fig.tight_layout()
plt.subplots_adjust(wspace=0.25)
plt.savefig(f'../../results/figs/filion_hs-{portrait}.pdf', dpi=300, bbox_inches='tight') #2010-{acnt_fits}-{cnt_fits}-fits

# %% [markdown]
# # 1b. Freezing, melting, condensation, cavitation

# %%
d0 = '../../results/lit/'
files = ['melting-7.4', 'sanchez_2020', 'menzl_2016', 'auer-frenkel-p15_'] #'sharma_2018',
labels = ['Melting - Gispen (2023)', 'Condensation - Sanchez-Burgos (2020)', 'Cavitation - Menzl (2016)', 'Freezing - Auer (2001)'] #'Anisotropic nucleus', 
dmus = [0.15,0.0537/0.6811, 135/4.092, 0.34]
rhos = [0.90,  0.6811, 1.0, 1.107]
order = [3,0,1,2]
files, labels, dmus, rhos = np.array(files)[order], np.array(labels)[order], np.array(dmus)[order], np.array(rhos)[order]

colors = ['C0', 'C1', 'C2', 'C3', 'C4']
colors = 5 * ['k']
portrait = True

nplots = 4
if portrait:
    fig, axs = plt.subplots(
        2,2, sharex=True, sharey=True, figsize=(3.375,1.4*3.375))
    axs = axs.flatten()
    plt.subplots_adjust(wspace=0.4, hspace=0.2)
else:
    fig, axs = plt.subplots(
        1,nplots, sharey=True, figsize=(2*3.375,3))

for i, f in enumerate(files):
    if f == 'sharma_2018':
        continue
    df = pd.read_csv(f'../../results/lit/{f}.csv')
    try:
        N, dG = df['N'], df['dG']
    except:
        V, dG = df['V'], df['dG']
        N = V
    acnt.rho_s = rho_s = rhos[i]
    acnt.dmu = dmu = dmus[i]
    Nc = N[np.argmax(dG)]
    order = np.argsort(N)
    N, dG = N[order], dG[order]
    mask = (N < Nc*2)
    axs[i].plot(N[mask]/Nc, dG[mask], '-', c=colors[i], label='US')

    # fit CNT from just critical nucleus size
    N_ = np.linspace(0, Nc*2, num=1000)
    axs[i].scatter(Nc/Nc, 0.5*Nc*dmu, c='C3')
    N_fit = [0, Nc]
    G_fit = [0, 0.5*Nc*dmu]
    params, cov = curve_fit(acnt.cnt, N_fit, G_fit, p0=[1,],)
    if plotx == 'r':
        axs[i].plot((N_/rho_s / (4*np.pi/3))**(1/3), acnt.cnt(N_, *params), '--', label=None, c='C3')
    else:
        axs[i].plot(N_/Nc, acnt.cnt(N_, *params), ':', label=None, c='C3', lw=2)
    axs[i].plot(Nc/Nc, 0.5*Nc*dmu, c='C3', label='CNT', ls=':', marker='o', lw=2)

    # fit aCNT
    mask = (dG > 10)
    N_fit, G_fit = N[mask], dG[mask]
    if f in ['menzl_2016']:
        params, cov = curve_fit(acnt.r_pol_g0, N_fit, G_fit, p0=[1, 0, 0])
        G2 = acnt.r_pol_g0(N_, *params)
    else:
        params, cov = curve_fit(acnt.r_pol_g0, N_fit, G_fit, p0=[1, 0, 0])
        G2 = acnt.r_pol_g0(N_, *params)
    axs[i].plot(N_/Nc, G2, '--', c='C0', label='aCNT')
    axs[i].text(1.0, 72, labels[i].split('-')[0][:-1], ha='center')

    axs[i].set_xlim(0, 2)
    axs[i].spines[['right', 'top']].set_visible(False)

if portrait:
    for iax, ax in enumerate(axs):
        if iax in [2,3]:
            ax.set_xlabel(r'$v ~/~ v_c$')
        if iax in [0, 2]:
            ax.set_ylabel(r'$\Delta G ~/~ k_B T$')
else:
    axs[0].set_xlabel(r'$v ~/~ v_c$')
    axs[0].xaxis.set_label_coords(2.3,-0.15)
axs[0].set_ylabel(r'$\Delta G ~/~ k_B T$')
plt.ylim(0, 70)
axs[-1].legend(ncol=1, loc='lower right')
plt.savefig(f'../../results/figs/melting-condensation-barrier-{portrait}.pdf', dpi=300, bbox_inches='tight')

# %% [markdown]
# # 2. Figures from Variational Umbrella Seeding results

# %% [markdown]
# ## 6a. WCA barriers

# %%
params_npt = [
    (4.32, -8.35, 13.46),
    (3.92, -5.56, 10.51),
    (3.64, -3.91, 9.39),
]

params_nve = [
    (4.31, -8.21, 13.22),
    (3.90, -5.33, 10.21),
    (3.50, -2.75, 7.99),
]

# %%
acnt.dmu = dmu = 0.41
acnt.rho_s = rho_s = 0.844

fig, axs = plt.subplots(
    1,3, sharey=True,
    gridspec_kw={'width_ratios':[1,1,1],}, figsize=(4,3)) #150,175,200
model, ys, fits_g0, skip_npt, skip_nve = 'wca', [6,7,8], 13.65, 50, 100
xmaxs = [200, 225, 250]

for integrator in ['npt','nve'][:1]:
    for iy, y in enumerate(ys[::-1]):

        # fit aCNT
        for ip, params_ in enumerate([params_npt, params_nve]):
            params = params_[::-1][iy]
            N = np.arange(0, 400)
            G = acnt.r_pol_g0(N, *params) 
            Nc = N[np.argmax(G)]
            mask = (N < 1.75*Nc)
            print(G.max())

            # plot aCNT
            ls = '--' #['-', '--', ':'][iy]
            label = 'VUS' if (iy == 2 and ip == 0) else None
            axs[iy].plot(N[mask], G[mask], ls=ls, label=label, c='C0')

            # fit CNT
            if ip == 0:
                N_fit = [0, Nc]
                G_fit = [0, 0.5*Nc*dmu]
                params, cov = curve_fit(acnt.cnt, N_fit, G_fit, p0=[1,],)
                label = 'CNT' if (iy == 2 and ip == 0) else None
                axs[iy].plot(N, acnt.cnt(N, *params), ':', lw=2, label=None, c='C3')
                axs[iy].plot(Nc, 0.5*Nc*dmu, ':', lw=2, label=label, c='C3', marker='o')
                # axs[iy].axvline(Nc)

        # axes and labels
        axs[iy].set_xticks([0, 100, 200, 300], labels=[0, None, 200, None])
        axs[iy].set_xlim(0, 300)
        axs[iy].set_ylim(0, 50)
        axs[iy].set_xlabel(rf'$n ~(\xi_c = {y})$')
        axs[iy].spines[['right', 'top']].set_visible(False)

axs[0].set_ylabel(r'$\Delta G ~/~ k_B T$')
axs[2].legend()
plt.savefig('../../results/figs/wca-barriers-npt-nve.pdf', dpi=300, bbox_inches='tight')

# %% [markdown]
# ## 6b. mW barriers

# %%
params_npt = [
    (0.62, 4.29, -3.47, 6.15),
    (0.50, 3.95, -0.06, 2.22),
    (0.38, 3.77, 2.69, -1.23),
]

params_nve = [
    (0.62, 4.25, -2.98, 5.43),
    (0.50, 3.99, -0.45, 2.74),
    (0.38, 3.77, 2.75, -1.32),
]

# %%
fig, ax = plt.subplots(figsize=(3.375, 3.375/4*3))
model, ys = 'mW', [215.1, 225.0, 235.0]
xmaxs = [200, 225, 250]

for integrator in ['npt','nve'][:1]:
    for iy, y in enumerate(ys[::-1]):
        # fit aCNT
        for ip, params_ in enumerate([params_npt, params_nve]):
            params = params_[::-1][iy]
            dmu = acnt.dmu = params[0]
            params = params[1:]
            N = np.arange(0, 400)
            G = acnt.r_pol_g0(N, *params) 
            Nc = N[np.argmax(G)]
            mask = (N < 1.75*Nc)

            # plot aCNT
            label = f'{y} K' if ip == 0 else None
            ls = ['-', '--', ':'][iy]
            ax.plot(N[mask], G[mask], ls=ls, label=label, c='C0')

# axes and labels
ax.set_xlim(0, 400)
ax.set_ylim(0, 80)
ax.set_xlabel(rf'$n$')
ax.spines[['right', 'top']].set_visible(False)

ax.set_ylabel(r'$\Delta G ~/~ k_B T$')
ax.legend()
plt.savefig('../../results/figs/mW-barriers-npt-nve.pdf', dpi=300, bbox_inches='tight')

# %% [markdown]
# ## 6c. TIP4P/ICE barrier

# %%
acnt.dmu = dmu = 0.3228
params = (4.32, -9.89, 16.02)

fig, ax = plt.subplots(figsize=(3.375, 3.375/4*3))
model, ys, fits_g0, skip_npt = 'tip4pice', [230.0], 10.0, 150

for integrator in ['npt','nve'][:1]:
    for iy, y in enumerate(ys[::-1]):

        # fit aCNT
        N = np.arange(3, 600)
        G = acnt.r_pol_g0(N, *params) 
        Nc = N[np.argmax(G)]
        Gc = G[np.argmax(G)]
        print(Nc, Gc)
        mask = (N < 1.75*Nc)
        ax.plot(Nc, Gc, marker='o', label='VUS')

        # plot aCNT
        label = f'{y} K'
        ls = ['-', '--', ':'][iy]
        ax.plot(N[mask], G[mask], ls=ls, label=None, c='C0')


ax.set_xlim(0, 600)
ax.set_ylim(0, 60)
ax.set_xlabel(rf'$n$')
ax.spines[['right', 'top']].set_visible(False)

ax.set_ylabel(r'$\Delta G ~/~ k_B T$')
ax.legend(loc='lower right')
plt.savefig('../../results/figs/tip4p-barrier.svg', dpi=300, bbox_inches='tight')

# %% [markdown]
# ## 6d. mW: nucleus sizes, barriers and nucleation rates

# %%
kB = 1.380648520000e-23  # Boltzmann's constant [m^2 kg s^-2 K^-1]
Nav = 6.02214090000e23

# %%
Tm = 274.6
sigma = 2.3925                                           # A
rho_s = 0.985e3/(18.0153e-3)*Nav*(sigma*1e-10)**3        # (/sigma^3)
Ts = np.array([215.1, 225.0, 235.0])    

plot_N, plot_gamma, plot_G, plot_rate = 0,1,2,3 #,1,2
nplots = max(plot_N, plot_gamma, plot_G, plot_rate)+1
fig, axs = plt.subplots(nplots, 1, figsize=(3.375, 6), sharex=True) #
colors = ['k', 'C3', 'C0',]
facecolors = ['k', 'C3', 'C0']
markers = ['o', 'o', 'o']
markersize = [10, 30, 30]
line_styles = ['-', ':', '--']
line_weights = [1, 2, 1]
zorders = [2, 1, 0]


# Collect US and Seeding 2.0 barriers
Ts = np.array([215.1, 225.0, 235.0])
J0s = np.array([4.8e39, 4.8e39, 1.2e40])
J0 = 5e39

dfs = {}
for i, key in enumerate(['US', 'Seeding', 'VUS',]):
    df = dfs[key] = pd.read_csv(f'../../results/lit/{key}_mW.csv', delim_whitespace=True, comment='#')
    if key == 'Seeding':
        pol_dmu = Polynomial.fit(Tm-df['dT'], df['dmu'], deg=1)


for i, key in enumerate(['US', 'Seeding', 'VUS',]):
    df = dfs[key]
    T = Tm - df['dT']
    axs[plot_N].scatter(Tm-T, df['Nc'], edgecolors=colors[i], facecolors=facecolors[i], marker=markers[i], s=markersize[i], zorder=zorders[i])
    axs[plot_G].scatter(Tm-T, df['Gc'], edgecolors=colors[i], facecolors=facecolors[i], marker=markers[i], s=markersize[i], zorder=zorders[i])
    
    # for legend
    axs[0].plot(T, -T, label=key, c=colors[i], lw=line_weights[i], ls=line_styles[i], marker='o')
    # axs[0].scatter(T, -T, label=key, linewidths=50, facecolors=facecolors[i], edgecolors=colors[i], lw=line_weights[i], ls=line_styles[i], marker='o')

    if 'log10J' in df:
        axs[plot_rate].scatter(Tm-T, df['log10J'], edgecolors=colors[i], facecolors=facecolors[i], marker=markers[i], s=markersize[i], zorder=zorders[i])
    else:
        axs[plot_rate].scatter(Tm-T,  np.log10(J0 * np.exp(-df['Gc'])), edgecolors=colors[i], facecolors=facecolors[i], marker=markers[i], s=markersize[i], zorder=zorders[i])

    if not 'gamma' in df:
        kT = kB*T*Nav / 4184                       # kcal/mol
        dmu = pol_dmu(Tm-df['dT'])/kT
        gamma_cf = 1e-3  / (kB*T) *(sigma*1e-10)**2
        # print(gamma_cf)
        df['gamma'] = (df['Gc'] *(3* dmu**2 * rho_s**2) / (16*np.pi))**(1/3) / gamma_cf
        # print(key, dmu, df['gamma'])
    
    T_ = np.linspace(200, 274, num=500)
    kT = kB*T_                                   # J
    kT_ = kB*T_*Nav / 4184                       # kcal/mol
    dmu = pol_dmu(T_)/kT_                        # kT 

    if 'gamma' in df:
        pol_gamma = Polynomial.fit(T, df['gamma'], deg=1)
        gamma_cf = 1e-3  / kT *(sigma*1e-10)**2     # conversion factor
        gamma_ = pol_gamma(T_) *gamma_cf            # kT/sigma^2
        Nc = 32*np.pi * gamma_**3 / (3* dmu**3 * rho_s**2)
        dG = 0.5*Nc*dmu

        if plot_N is not False:
            if key == 'Seeding':
                axs[plot_N].plot(Tm-T_, Nc, c=colors[i], ls=line_styles[i], lw=line_weights[i], zorder=zorders[i])
                axs[plot_N].fill_between(Tm-T_, Nc-Nc**(2/3), Nc+Nc**(2/3), zorder=zorders[i], alpha=0.1, color=colors[i])
        if plot_G:
            axs[plot_G].plot(Tm-T_, dG, c=colors[i], ls=line_styles[i], lw=line_weights[i])
            if key == 'Seeding':
                dGmin = 0.5*(Nc-Nc**(2/3))*dmu
                dGmax = 0.5*(Nc+Nc**(2/3))*dmu
                axs[plot_G].fill_between(Tm-T_, dGmin, dGmax, zorder=-1, alpha=0.1, color=colors[i])
        if plot_gamma:
            axs[plot_gamma].plot(Tm-T_, pol_gamma(T_), c=colors[i], ls=line_styles[i], lw=line_weights[i], zorder=zorders[i])
            axs[plot_gamma].scatter(Tm-T, df['gamma'], edgecolors=colors[i], facecolors=facecolors[i], marker=markers[i], s=markersize[i], zorder=zorders[i])
            if key == 'Seeding':
                gamma_min = ((Nc-Nc**(2/3)) *(3* dmu**3 * rho_s**2) / (32*np.pi))**(1/3) / gamma_cf
                gamma_max = ((Nc+Nc**(2/3)) *(3* dmu**3 * rho_s**2) / (32*np.pi))**(1/3) / gamma_cf
                axs[plot_gamma].fill_between(Tm-T_, gamma_min, gamma_max, zorder=zorders[i], alpha=0.1, color=colors[i])
        if plot_rate:
            axs[plot_rate].plot(Tm-T_, np.log10(J0 * np.exp(-dG)), c=colors[i], ls=line_styles[i], lw=line_weights[i], zorder=zorders[i])
            if key == 'Seeding':
                rate_min = np.log10(J0 * np.exp(-dGmax))
                rate_max = np.log10(J0 * np.exp(-dGmin))
                axs[plot_rate].fill_between(Tm-T_, rate_min, rate_max, zorder=zorders[i], alpha=0.1, color=colors[i])


### TWEAK APPEARANCE
# Nc axis
axs[plot_N].set_ylabel(r'$n_c$')
axs[plot_N].set_yticks(np.arange(0, 1000, 200))
axs[plot_N].set_ylim(0, 800)
axs[plot_N].grid(axis='y', zorder=0, which='both')

# gamma axis
if plot_gamma:
    axs[plot_gamma].set_ylabel(r'$\gamma~ \mathrm{(mJ/m^2)}$')
    axs[plot_gamma].set_ylim(20, 40)
    # axs[plot_gamma].set_yticks(np.arange(0, 120, 20))
    axs[plot_gamma].grid(axis='y', zorder=0, which='both')

# dG axis
axs[plot_G].set_ylabel(r'$\Delta G^c / ~k_B T$')
axs[plot_G].set_ylim(0, 100)
axs[plot_G].set_yticks(np.arange(0, 160, 40))
axs[plot_G].grid(axis='y', zorder=0, which='both')

# rate axis
axs[plot_rate].set_ylabel(r'$\log (J ~ \mathrm{m^3 s})$')
axs[plot_rate].set_xlabel(r'$\Delta T ~/~ \mathrm{K}$')
axs[plot_rate].set_xlim(30, 70) #212, 238)
axs[plot_rate].set_yticks(np.arange(-20, 60, 20))
axs[plot_rate].set_ylim(-20, 40,)

axs[plot_rate].grid(axis='y', zorder=-1, which='both')
for ax in axs:
    ax.spines[['right', 'top']].set_visible(False)


handles, labels = axs[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), ncol=3, title='', bbox_to_anchor=(0,0,0.92,0.95),)
plt.savefig(f'../../results/figs/mW-order-params-cnt.pdf', dpi=300, bbox_inches='tight')

# %% [markdown]
# ## 6d. TIP4P/ICE: nucleation rates

# %%
plot_N, plot_gamma, plot_G, plot_rate = False,False,False,0
nplots = max(plot_N, plot_gamma, plot_G, plot_rate)+1
fig, axs = plt.subplots(nplots, 1, figsize=(3.375, 3.375/4*3), sharex=True)
if nplots == 1:
    axs = [axs]
colors = ['k', 'C3', 'C0',]
line_styles = ['-', ':', '--']
line_weights = [1, 2, 1]


dfs = {}
for i, key in enumerate(['MTD', 'Seeding', 'VUS',]):
    df = dfs[key] = pd.read_csv(f'../../results/lit/{key}_tip4p.csv', delim_whitespace=True, comment='#')
    if key == 'seeding':
        pol_dmu = Polynomial.fit(Tm-df['dT'], df['dmu'], deg=1)

Tm = 270
T_ = np.linspace(200, Tm, num=2000)

# from "On the time required" and "Interfacial free energy as the key" (both Espinosa 2016)
df = dfs['Seeding']
if plot_gamma is not False:
    axs[plot_gamma].scatter(Tm-df['T'], df['gamma'], c=colors[1])
gamma_ = 30.044 - 0.27477 *(270 - T_)
gamma_min = 28.5 - 0.306 *(270 - T_)
gamma_max = 31.3 - 0.234 *(270 - T_)
if plot_gamma is not False:
    axs[plot_gamma].plot(Tm-T_, gamma_, c=colors[1], ls='--')
    axs[plot_gamma].fill_between(Tm-T_, gamma_min, gamma_max, color=colors[1], alpha=0.1)
    axs[plot_gamma].scatter(0, 29.8, label='Lattice Mold', marker='D', c='C4', zorder=1e9)
if plot_rate is not False:
    axs[plot_rate].scatter(Tm-230, 14.1, label='LM', marker='D', c='C4')

# CNT fits
kT = kB*T_                                      # J
kT_ = kB*T_*Nav / 4184                          # kcal/mol
dmu = 0.146/kT_[np.argmin(np.abs(T_-230))] /(270-230)*(270 - T_)           # kT (from "On the time required")
sigma = 3.1668e-10                              # m
rho_s = 0.9112e3/(18.0154e-3)*Nav*sigma**3      # "Lattice mold 2022" (/sigma^3)
J0 = 1e37                                       # "Homogeneous ice nucleation evaluated for several water models"

# Scatter seeding rates
for i, T in enumerate(dfs['Seeding']['T']):
    Nc = dfs['Seeding']['Nc'][i]
    dG = 0.5*Nc*dmu[np.argmin(np.abs(T - T_))]
    label = 'Seeding' if i == 0 else None
    axs[plot_rate].scatter(Tm-T, np.log10(J0*np.exp(-dG)), c=colors[1])

# Estimate error from gamma
rates = []
for i, gamma in enumerate([gamma_, gamma_min, gamma_max]):
    gamma = 1e-3*gamma/kT*sigma**2                      # (kT/sigma^2) "On the time required"
    Gc = 16*np.pi * gamma**3 / (3*rho_s**2 * dmu**2)    # (kT)
    rates.append(np.log10(J0*np.exp(-Gc)))
mask = np.isfinite(rates[1])*np.isfinite(rates[2])*np.isfinite(rates[0]) * (Tm-T_ > 20)
error_up = Polynomial.fit(Tm-T_[mask], (rates[1]-rates[0])[mask], deg=4)
error_down = Polynomial.fit(Tm-T_[mask], (rates[2]-rates[0])[mask], deg=4)

# Plot extrapolated rate from "On the time required" (2016)
df = pd.read_csv('../../results/lit/seeding_tip4p_rates_.csv')
axs[plot_rate].plot(df['dT'], df['log10J'], c=colors[1], ls='--',)
axs[plot_rate].plot(-df['dT'], df['log10J'], c=colors[1], ls='--', marker='o', label='Seeding')
axs[plot_rate].fill_between(
    df['dT'],
    df['log10J']-5,
    df['log10J']+5,
    color=colors[1], ls='--', alpha=0.1)

## Metadynamics from Niu 2019
df_niu = pd.read_csv('../../results/lit/MTD_tip4p.csv', comment='#')
axs[plot_rate].plot(df_niu['dT'], df_niu['log10J'], c='k', label='MTD')

# Variational Umbrella Seeding
Gc = 50.4
T_ = 230
kT_ = kB*T_*Nav / 4184
kT = kB*T_
dmu = 0.146/kT_
gamma = (Gc * (3*rho_s**2 * dmu**2) / (16*np.pi))**(1/3)
gamma = gamma / (1e-3/kT*sigma**2)
if plot_gamma is not False:
    axs[plot_gamma].scatter(Tm-T_, gamma, zorder=1e9, c=colors[2])
if plot_rate is not False:
    label = 'Variational Umbrella Seeding'
    axs[plot_rate].scatter(Tm-T_, np.log10(J0*np.exp(-Gc)), zorder=1e9, c=colors[2], label='VUS')


### TWEAK APPEARANCE
# Nc axis
if plot_N is not False:
    axs[plot_N].set_ylabel(r'$n^*$')
    axs[plot_N].set_yticks(np.arange(0, 1000, 200))
    axs[plot_N].set_ylim(0, 800)
    axs[plot_N].grid(axis='y', zorder=0, which='both')

# gamma axis
if plot_gamma is not False:
    axs[plot_gamma].set_ylabel(r'$\gamma~ \mathrm{(mJ/m^2)}$')
    axs[plot_gamma].set_ylim(0, 40)
    axs[plot_gamma].grid(axis='y', zorder=0, which='both')

# dG axis
if plot_G is not False:
    axs[plot_G].set_ylabel(r'$\Delta G* / ~k_B T$')
    axs[plot_G].set_ylim(0, 100)
    axs[plot_G].set_yticks(np.arange(0, 160, 40))
    axs[plot_G].grid(axis='y', zorder=0, which='both')

# rate axis
if plot_rate is not False:
    axs[plot_rate].set_ylabel(r'$\log (J ~ \mathrm{m^3 s})$')
    axs[plot_rate].set_xlabel(r'$\Delta T ~/~ \mathrm{K}$')
    if plot_gamma is False:
        axs[plot_rate].set_xlim(28, 57)
        axs[plot_rate].set_ylim(-20, 40,)
    else:
        axs[plot_rate].set_xlim(0, 70)
        axs[plot_rate].set_ylim(-40, 40,)
        axs[plot_rate].set_yticks(np.arange(-40, 60, 20))
    axs[plot_rate].grid(axis='y', zorder=-1, which='both')


handles, labels = axs[plot_rate].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axs[plot_rate].legend(by_label.values(), by_label.keys(), ncol=1)
axs[plot_rate].spines[['right', 'top']].set_visible(False)
plt.savefig(f'../../results/figs/tip4pice_rate.svg', dpi=300, bbox_inches='tight')


