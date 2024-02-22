# This script defines aCNT functional forms

def cnt(N, gamma):
    import numpy as np
    R = (N/rho_s / (4*np.pi/3))**(1/3)
    G = -dmu *rho_s * (4*np.pi/3)*  R**3
    G += 4*np.pi * gamma * R**2
    return G

def get_cnt(gamma):
    def fun(N):
        return cnt(N, gamma)
    return fun

def r_pol(N, g2, g1):
    G = -dmu * N
    G += g1*N**(1/3) + g2*N**(2/3)
    return G

def r_pol_g0(N, g2, g1, g0):
    G = -dmu * N
    G += g1*N**(1/3) + g2*N**(2/3)
    G += g0
    return G

def r_pol_max(N, *params):
    return -r_pol(N, *params)

def r_pol_g0_max(N, *params):
    return -r_pol_g0(N, *params)
