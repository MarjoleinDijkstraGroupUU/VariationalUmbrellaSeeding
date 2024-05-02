# %%
# %load_ext autoreload
# %autoreload 2

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from natsort import natsorted
import os
import warnings
from scipy.special import logsumexp
warnings.filterwarnings('ignore')

# %%
import acnt

# %% [markdown]
# # 1. Define fitting methods

# %%
# plot local biased probability distribution and local free-energy profile
def plot_local_free_energy(G_bias_, target_size, params, N, p, plot_g0, dU, color0, i):
    plot_mask = (G_bias_ - G_bias_.min()) < 4 #10.0
    pmin = 0
    plot_mask = (p > pmin)
    G1 = acnt.r_pol(target_size, *params) -np.log(p[plot_mask]) + np.log(p.max()) + plot_g0
    plot_mask = ((G1 - G1.min()) < 4)
    plt.plot(N[p>pmin][plot_mask], G1[plot_mask] - dU[p>pmin][plot_mask], c=f'C{color0+i}', ls=':', zorder=2e9)
    plt.plot(N[p>pmin][plot_mask], G1[plot_mask], c=f'C{color0+i}', ls='--')


# compute likelihood for given set of params
# and make some plots of the barrier if requested
def compute_negative_loglikelihood(params, windows, Ns, p_s, plot_g0=None, color0=0):
    logL = 0 # total loglikelihood

    # iterate over simulation data
    for i, (target_size, bias_width) in enumerate(windows):
        if target_size == 0:
            continue

        # read samples
        nmin = 5
        mask = (Ns[i] > nmin)
        N = Ns[i][mask] # nucleus size
        p = p_s[i][mask] # nucleus size distribution

        # compute local likelihood
        bias_width_ = bias_width if target_size > 0 else np.inf
        dU = 0.5 * (N - target_size)**2 / bias_width_**2 # bias potential
        G = acnt.r_pol(N, *params) # local free-energy according to aCNT parameters
        G_bias = G + dU

        # compute normalization
        if target_size != 0:
            N_ = np.arange(max(nmin, target_size-10*bias_width), target_size+10*bias_width)
        else:
            N_ = np.arange(nmin, 30)
        G_ = acnt.r_pol(N_, *params)
        dU_ = 0.5 * (N_ - target_size)**2 / bias_width_**2
        G_bias_ = G_ + dU_
        logZ = logsumexp(-G_bias_) # log of normalization constant
        G_bias += logZ

        # add local loglikelihood to total loglikelihood
        logL -= np.sum(-p * G_bias)

        # If requested, plot local biased probability distribution and local free-energy profile
        if plot_g0 is not None:
            plot_local_free_energy(G_bias_, target_size, params, N, p, plot_g0, dU, color0, i)

    return logL


# Fit the constant additive parameter 'g_0' of the aCNT
def fit_g0(params, windows, Ns, p_s, plot=True):
    # get unbiased part
    for i, (target_size, bias_width) in enumerate(windows):
        if target_size == 0:
            # get unbiased nucleus size distribution
            mask = (p_s[i] > 0)
            Nu = Ns[i][mask]
            p_ = p_s[i][mask]

            # convert to free-energy
            Gu = -np.log(p_).values

            # subtract log(total number of particles)
            Ntot = p_[0]
            for k in Nu:
                Ntot += k * p_[k]
            Gu -= -np.log(Ntot)
            break

    # fit g0 by glueing unbiased free-energy Gu to acnt fit
    iNG0 = np.argmin(np.abs(Gu - 10.0))
    N0 = Nu[iNG0]
    g0 = Gu[iNG0] - acnt.r_pol(N0, *params)
    mask = (Gu < 11)
    if plot:
        plt.plot(Nu[mask], Gu[mask], c='k', zorder=1e8, label=r'$-\log p(n)$', lw=1)

    return g0


# Fit the aCNT parameters to observations
def fit_acnt_mle(windows, Ns, p_s, plot=True):

    # maximize loglikehood to optimize aCNT params
    bound = 100
    params = minimize(
        compute_negative_loglikelihood,
        args=(windows, Ns, p_s),
        x0=(0.5, 0.0,), bounds=((-bound, bound), (-bound, bound),)
        ).x
    
    # fit the constant additive parameter 'g_0' of the aCNT
    g0 = fit_g0(params, windows, Ns, p_s, plot)
    params = np.append(params, g0)

    # compute critical nucleus size and barrier height from aCNT parameters
    sol = minimize(acnt.r_pol_g0_max, args=tuple(params), x0=100)
    Nc, Gc = round(sol.x[0]), -sol.fun

    return Nc, Gc, params

# %% [markdown]
# # 2. Get simulation data

# %%
# define thermodynamic quantities and simulation model
def get_model(model, y):
    if model == 'wca':
        p = 12.0 # pressure
        bonds = y # number of solidlike bonds as solid-fluid criterion
        d0 = f'../../results/hmc/{model}/p{p}_{bonds}-bonds' # base directory of data
        dmu = 0.41 # supersaturation
        rho_s = 0.844 # density of solid phase
        kT = 1.0 
        sigma = 1.0 # diameter of a particle
    else:
        T = y # temperature
        kB = 1.380648520000e-23  # Boltzmann's constant [m^2 kg s^-2 K^-1]
        Nav = 6.02214090000e23

        if model == 'mW':
            kT = kB*T               # J
            kT_ = kB*T*Nav / 4184   # kcal/mol
            sigma = 2.3925          # diameter of a particle, in Angstrom
            rho_s = 0.985e3/(18.0153e-3)*Nav*(sigma*1e-10)**3        # (/sigma^3)
            dmu = -0.1553/kT_ / 34.6 *(T-274.6) # supersaturation in kT, taken from "Homogeneous ice nucleation evaluated for several water models (2014)"
            d0 = f'../../results/hmc/{model}/T{T}'
        elif model == 'tip4pice':
            kT = kB*T               # J
            kT_ = kB*T*Nav / 4184   # kcal/mol
            dmu = 0.146/kT_ /(270-230)*(270 - T) #("On the time required") kT
            sigma = 3.1668e-10          # diameter of a particle, in meter
            rho_s = 0.9112e3/(18.0154e-3)*Nav*sigma**3 # "Lattice mold 2022" (/sigma^3)
            gamma = 1e-3*(30.044 - 0.27477*(270 - T))/kT*sigma**2 # (kT/sigma^2) "On the time required"
            d0 = f'../../results/hmc/{model}/T{T}' # base directory of data

    return d0, dmu


# read logs of nucleus size over time
def get_data():
    Ns = [] # nucleus size
    p_s = [] # nucleus size distribution
    windows = [] # target sizes and bias widths

    # compute autocorrelation time of a correlated timeseries
    def compute_acf(y):
        def autocorrelation(data, lag):
            n = len(data)
            mean = np.mean(data)
            numerator = np.sum((data[:n-lag] - mean) * (data[lag:] - mean))
            denominator = n * np.var(data)
            acf = numerator / denominator
            return acf

        # Calculate the autocorrelation for a range of lags
        max_lag = len(y)//2
        lags = np.arange(1, max_lag, 10)
        autocorrelation_values = np.array([autocorrelation(y, round(lag)) for lag in lags])
        threshold = (1/np.e)
        auto_correlation_time = lags[np.argmin(np.abs(autocorrelation_values-threshold))]
        
        return auto_correlation_time

    # iterate over target sizes
    for n_ in natsorted(os.listdir(d0)):
        if not n_.startswith('n'):
            continue
        n = round(float(n_[1:])) # target size
        df = None # empty pandas dataframe

        # find right subdirectory for data
        integrator_ = integrator if n > 0 else 'npt'
        if n == 0:
            d2_ = [f for f in os.listdir(f'{d0}/{n_}') if f.startswith(integrator_) and f.endswith('False')][0]
        else:
            d2_ = [f for f in os.listdir(f'{d0}/{n_}') if f.startswith(integrator_)][0]
        d2 = f'{d0}/{n_}/{d2_}'
        
        # iterate over all simulations for one target size
        for i in os.listdir(d2):

            # find last nucleus size histogram
            d1 = f'{d0}/{n_}/{d2_}/{i}'
            hists = [f for f in os.listdir(d1) if f.startswith('nucleus_size')]
            files = natsorted(hists)

            # find bias width
            if n != 0:
                args = pd.read_csv(f'{d1}/args.dat', delim_whitespace=True, header=None, names=['key', 'value'], comment='#')
                bias_width = float(args[args['key'] == 'bias_width']['value'])
            else:
                bias_width = np.inf

            # read nucleus size distribution
            nucleus_size_hist_path = f'{d1}/nucleus_size_hist_processed.csv'
            if os.path.exists(nucleus_size_hist_path):
                df1 = pd.read_csv(nucleus_size_hist_path,)
            elif n == 0:
                df1 = pd.read_csv(f'{d1}/{files[-1]}',)
                files = files[len(files)//2:]
                if len(files) > 1:
                    df0 = pd.read_csv(f'{d1}/{files[0]}',)
                    df1['count'] -= df0['count']
            else:
                # read nucleus size timeseries
                thermo = pd.read_csv(f'{d1}/thermo.dat', delim_whitespace=True, comment='#')

                # estimate correlation length and subsample
                observable = thermo['nucleus_size'].values
                observable = observable[round(len(thermo)/10):]
                skip = round(compute_acf(observable))
                thermo = thermo[len(thermo)//10::skip]
                            
                # obtain nucleus size distribution
                nmin = round(max(0, n-20*bias_width))
                nmax = round(n+20*bias_width)
                bins = np.arange(nmin, nmax+2) - 0.5
                N = np.arange(nmin, nmax+1)
                p, _ = np.histogram(thermo['nucleus_size'], bins=bins)
                df1 = pd.DataFrame(np.array([N, p]).T, columns=['nucleus_size', 'count'])

            # accumulate
            df1.to_csv(nucleus_size_hist_path, index=False)
            if df is None:
                df = df1.__deepcopy__()
            else:
                df['count'] += df1['count']

        # gather data to return
        if df is not None:
            windows.append((n, bias_width,))
            Ns.append(df['nucleus_size'])
            p_s.append(df['count'])

    return windows, Ns, p_s

# %% [markdown]
# # Confidence interval

# %%
# estimate confidence intervals for the nucleus size and barrier height
def get_confidence_interval(params, windows, Ns, p_s):
    # compute loglikelihood of most likely parameters
    logL0 = compute_negative_loglikelihood(params[:-1], windows, Ns, p_s, plot_g0=None)

    # compute loglikelihood test threshold
    coverage = 0.95 # 95% confidence interval
    df = 2 # number of parameters to estimate
    dlogL0 = chi2.ppf(coverage, df=df) / 2
    Lrel0 = np.exp(-dlogL0)

    # compute loglikelihood relative to most likely parameters
    def get_dL(params_):
        logL = compute_negative_loglikelihood(params_, windows, Ns, p_s, plot_g0=None)
        return logL0 - logL

    # compute critical nucleus size and barrier height
    def get_Nc_G(params_):
        # refit g0
        g0 = fit_g0(params_, windows, Ns, p_s, plot=False)
        params_.append(g0)

        # compute Nc and dG
        sol = minimize(acnt.r_pol_g0_max, args=tuple(params_), x0=100)
        Nc, dG = round(sol.x[0]), -sol.fun

        return Nc, dG, params_

    # plot Nc and G versus relative likelihood
    def plot_intermediate(params_, plot='G'):
        Nc, G, params_ = get_Nc_G(params_)
        dlogL = get_dL(params_[:-1])
        Lrel = np.exp(dlogL)

        if Lrel < 1:
            # plot Nc and G versus relative likelihood
            if plot == 'G':
                plt.scatter(G, Lrel, c='C0')
            else:
                plt.scatter(Nc, Lrel, c='C0')

            # print Nc and G if relative likelihood is close to boundary
            if np.abs(Lrel-Lrel0)<5e-3:
                print(Nc, G, Lrel)

    # vary aCNT parameters and compute relative likelihood
    for dg2 in np.linspace(-0.2, 0.2, num=200):
        for dg1 in np.linspace(-0.2, 0.2, num=3):
            params_ = [params[0]+dg2, params[1]+dg1]
            plot_intermediate(params_, plot='G')
    plt.ylim(0, 1)
    plt.show()

# %% [markdown]
# # 3. Fit barriers

# %%
model, ys = 'wca', [6,7,8]
# model, ys = 'mW', [215.1, 225.0, 235.0],
# model, ys = 'tip4pice', [230.0]
plot = True
confidence = False

for integrator in ['npt','nve'][:]:
    for y in ys[:]:
        print(y)
        # get data
        d0, acnt.dmu = get_model(model, y)
        windows, Ns, p_s = get_data()

        # fit aCNT
        Nc, Gc, params = fit_acnt_mle(windows, Ns, p_s, plot=plot)

        print(f'Supersaturation (|Δμ|):  {acnt.dmu:.3f} kT')
        print('aCNT fit parameters (g2, g1, g0):', params)
        print('Integrator:', integrator)
        print('Barrier height:', f'{Gc:.1f} kT')
        print('Critical nucleus size (n*):', Nc)
        print('n* |Δμ| / 2:', f'{0.5*Nc*acnt.dmu:.1f} kT')
        # print(acnt.dmu, integrator, y, params, f'{Gc:.1f} {0.5*Nc*acnt.dmu:.1f} {Nc}')

        if plot:
            # plot local free-energy profiles
            compute_negative_loglikelihood(params[:-1], windows, Ns, p_s, plot_g0=params[-1])

            # plot fitted total free-energy profile
            N = np.arange(0, 1.2*Nc)
            G = acnt.r_pol_g0(N, *params)
            N0_ = np.argmin(np.abs(G-10.0))
            mask = (N > N0_)
            plt.plot(N[mask], G[mask], c='k', label=r'$\beta \Delta G_{\mathrm{aCNT}} (\gamma,\alpha,\Delta)$')

            # savefig
            plt.xlabel(r'$n$')
            plt.ylabel(r'$\beta \Delta G$')
            plt.xlim(0,)
            plt.ylim(0,)
            plt.legend(loc='lower right')
            # plt.savefig(f'../../results/figs/{model}-{y}-barrier.png', dpi=300)
            plt.show()

        if confidence:
            get_confidence_interval(params, windows, Ns, p_s,)    



# %%



