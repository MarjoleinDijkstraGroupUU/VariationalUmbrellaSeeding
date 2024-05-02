# %%
# %load_ext autoreload
# %autoreload 2

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import warnings
from scipy.special import logsumexp
warnings.filterwarnings('ignore')

# %%
import acnt

# %% [markdown]
# ## 1. Compute bias widths for umbrella sampling simulations

# %%
model = 'wca'

if model == 'wca':
    T = 1.0
    kT = 1.0
    dmu =  0.41
    rho_nucleus = rho_s = 0.844
    t_Nc = [(5.7, 185), (6.7, 155), (7.7, 130)]
else:
    kB = 1.380648520000e-23  # Boltzmann's constant [m^2 kg s^-2 K^-1]
    Nav = 6.02214090000e23
    if model == 'mW':
        T = 225.0 #.1
        kT = kB * T * Nav / 4184
        rho_nucleus = rho_s = 0.451
        dmu = 0.216/kT /(225-274.6)*(T-274.6)
        t_Nc = {'215.1': [(None, 70),], '225.0': [(None, 150)], '235.0': [(None, 320)]}[f'{T:.1f}']
    elif model == 'tip4pice':
        T = 230.0
        kT = kB * T * Nav / 4184
        rho_nucleus = rho_s = 0.9673
        dmu = 0.319 # kT
        t_Nc = [(None, 240)]

acnt.dmu = dmu
acnt.rho_s = rho_s

for threshold, Nc in t_Nc:
    dG = 0.5 * Nc * dmu
    print(dG, Nc, dmu)
    gamma = (2 * dG * dmu**2 * rho_nucleus**2 * 3/(32 * np.pi))**(1/3)
    print(gamma)

    # get CNT free-energy profile and compute critical nucleus size
    free_energy = acnt.get_cnt(gamma)
    def neg_free_energy(N):
        return -free_energy(N)
    NcGc = minimize(neg_free_energy, x0=100)
    Nc, Gc = NcGc.x[0], -NcGc.fun
    
    # compute bias width from local curvature
    windows = [(0, 1)]
    num_windows = 4
    N_min = max(10, round(Nc/num_windows))
    for i in range(num_windows):
        target_size = round(N_min + i*(Nc - N_min)/(num_windows-1))
        curv = (free_energy(target_size+1,) + free_energy(target_size-1) - 2*free_energy(target_size))
        bias_width = 1 / np.sqrt(np.abs(6*curv))
        windows.append((target_size, bias_width))

    # print bias widths for bash submission
    for target_size, bias_width in windows:
        for integrator in ["npt"]: #, "nve"]:
            print(f' \"{T} {target_size} {integrator} {bias_width:.2f}\"', end="")

    # plot to illustrate windows
    N = np.arange(0, 1.5*Nc)
    G = free_energy(N,)
    plt.plot(N, G, c='k')
    
    for i, (target_size, bias_width) in enumerate(windows):
        dU = 0.5 * (N - target_size)**2 / bias_width**2 / kT
        G_bias = G + dU

        plot_mask = (G_bias - G_bias.min()) > 0 #< 20.0
        if target_size != 0:
            plt.plot(N[plot_mask], G_bias[plot_mask], c='C0')
    
    plt.xlim(0, Nc*1.5)
    plt.ylim(0, Gc*1.2)
    plt.legend()
    plt.show()

# %% [markdown]
# # 2. Define fitting methods

# %%

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
            Gu -= -np.log(p_[0] + p_[1] + 2*p_[2] + 3*p_[3] + 4*p_[4])
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
    try:
        g0 = fit_g0(params, windows, Ns, p_s, plot)
    except:
        g0 = 0
    params = np.append(params, g0)

    # compute critical nucleus size and barrier height from aCNT parameters
    sol = minimize(acnt.r_pol_g0_max, args=tuple(params), x0=100)
    Nc, Gc = round(sol.x[0]), -sol.fun

    return Nc, Gc, params

# %% [markdown]
# # 3. Sample from ideal free-energy profile

# %%
from numpy.random import default_rng
rng = default_rng()

# %%
# draw total_samples independent nucleus sizes from the free-energy profile
# with bias potentials defined by 'windows'
def sample(free_energy, windows, total_samples):
    def neg_free_energy(N):
        return -free_energy(N)
    NcGc = minimize(neg_free_energy, x0=100)
    Nc, Gc = NcGc.x[0], -NcGc.fun

    # sample from target free energy
    Ns = []
    p_s = []
    dUs = []

    # iterate over target sizes
    for i, (target_size, bias_width) in enumerate(windows):
        # compute distribution with umbrella bias
        N_local = np.arange(round(target_size-10*bias_width), round(target_size+10*bias_width))
        N_local = N_local[N_local > 0]
        dU = 0.5 * (N_local - target_size)**2 / bias_width**2
        dUs.append(dU[None, :])
        G = free_energy(N_local)
        logp = (G+dU)
        p = np.exp(-logp + logp[np.argmin(np.abs(dU))])
        
        # sample
        n_samples = total_samples // len(windows)
        x = rng.choice(N_local, size=n_samples, p=p/p.sum())
        p_, _ = np.histogram(x, range=(N_local.min()-0.5,N_local.max()+0.5), bins=len(N_local), density=False)
        Ns.append(N_local)
        p_s.append(p_)

    return Ns, p_s, dUs

# %% [markdown]
# # 4. Try different sampling strategies

# %%
rmsd = {} # errors

acnt.rho_s = 1.0
kT = 1.0 # thermal energy
gamma = 0.7 # interfacial tension
repeats = 500 # repeats for one sampling strategy
n_samples = 50 # total number of independent samples from free-energy profile
num_windows = 4 #[2,3,4,5,6,8,10] # number of different target sizes
spring_constant_mode = 'vary' #['max', 'vary', 'top']
ys = [20, 40, 60, 80, 100]

for y in ys:
    n_samples = y
    # num_windows = y
    # mode = y
    # mode, n_samples = y.split(';')
    # n_samples = float(n_samples)
    # spring_constant_mode = y
    
    rmsd[f'{y}'] = []
    print(f'y = {y}')
    for dmu in [0.3]:
        acnt.dmu = dmu
        errors = []

        # get free-energy profile
        free_energy = acnt.get_cnt(gamma)
        def neg_free_energy(N):
            return -free_energy(N)
        # compute critical nucleus size, barrier height (with respect to N_min), and curvature at the top
        NcGc = minimize(neg_free_energy, x0=100)
        Nc, Gc = NcGc.x[0], -NcGc.fun
        dG2dN2 = dmu / (3 * Nc)
        N_min = 10
        Gc -= free_energy(N_min)
        
        # define windows, i.e. target sizes and bias widths
        windows = []
        for i in range(num_windows):
            # distribute target sizes uniformly between 0 and critical nucleus size
            N_min_ = max(N_min, round(Nc/num_windows))
            N_max_ = round(Nc)
            if num_windows == 1:
                target_size = Nc
            else:
                target_size = round(N_min_ + i*(Nc - N_min_)/(num_windows-1))

            # specify bias potential width
            if type(spring_constant_mode) == float:
                bias_width = spring_constant_mode
            else:
                # compute curvature of free-energy profile:
                if spring_constant_mode == 'vary':
                    # local curvature
                    curv = (free_energy(target_size+1,) + free_energy(target_size-1) - 2*free_energy(target_size))
                elif spring_constant_mode == 'max':
                    # maximum local curvature, i.e. at the smallest target size
                    curv = (free_energy(N_min_+1,) + free_energy(N_min_-1) - 2*free_energy(N_min_))
                elif spring_constant_mode == 'top':
                    # curvature at the top of the barrier i.e. at the critical nucleus size
                    curv = (free_energy(Nc+1,) + free_energy(Nc-1) - 2*free_energy(Nc))
                # compute bias potential width from curvature
                k = 6*np.abs(curv)
                bias_width = 1 / np.sqrt(k)
            
            windows.append((target_size, bias_width))
            
        # sample and fit a few times
        for j in range(repeats):
            # sample
            Ns, p_s, dUs = sample(free_energy, windows, int(n_samples))

            # compute barrier height according to aCNT fit
            Nc, Gc_, params = fit_acnt_mle(windows, Ns, p_s)
            Gc_ = Gc_ - acnt.r_pol(N_min, *params[:-1])

            # record error
            errors.append(Gc - Gc_)
            if j == 0:
                print(Gc, Gc_)

        # record average errors
        errors = np.array(errors)
        rmsd[f'{y}'].append((Gc, np.mean(errors), np.sqrt(np.mean(errors**2))))

# %%
# plot error vs sampling strategy
print(rmsd)
for i, y in enumerate(rmsd):
    dGerr = np.array(rmsd[y])
    dG, bias, err = dGerr[:, 0], dGerr[:, 1], dGerr[:, 2]
    plt.scatter(dG, err, label=y, c=f'C{i}', zorder=1e9-i)

# labels and appearance
plt.ylabel('RMS error (kT)')
plt.xlabel(r'$\beta \Delta G^*$')
plt.xlim(0, )
plt.ylim(0, )

# remove duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),title='Fit strategy', loc='upper left')
plt.show()

# %%
# plot to illustrate windows
N = np.arange(0, 1.5*Nc)
G = free_energy(N,)
plt.plot(N, G, c='k')

for i, (target_size, bias_width) in enumerate(windows):
    print(target_size, bias_width)
    dU = 0.5 * (N - target_size)**2 / bias_width**2
    G_bias = G + dU

    plot_mask = (G_bias - G_bias.min()) < 20.0
    if target_size != 0:
        plt.plot(N[plot_mask], G_bias[plot_mask], c='C0')

plt.xlim(0, )
plt.ylim(0, )

# %%



