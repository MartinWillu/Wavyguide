import sys
import os
sys.path.append("C:\\Users\\kmari\\OneDrive\\Documents\\Thesis\\Wavyguide-main\\modesolverpyfolder")

import numpy as np
import opticalmaterialspy as mat
from modesolverpy.structure import RidgeWaveguide_TE
from modesolverpy.mode_solver import ModeSolverFullyVectorial
from modesolverpy.structure_base import StructureAni
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import PchipInterpolator

c = 2.998e8 # m/s
d33 = 27e-12 # m/V
deff = d33/np.pi
epsilon_0 = 8.854e-12 # F/m
Z_0 = 1/epsilon_0/c

rng = np.random.default_rng()

# alpha = 1 or 2 dB/cm
# 20 log_10(e^(- alpha * 1 cm)) = -1
# e^(- alpha * 1 cm) = 10^(-1/20)
# - alpha*1cm = ln(10^(-0.05))
# alpha per cm = 0.115
alpha = 0 # 1/um



wl_min = 0.8 # um
fr_max = c/(wl_min*1e-6) # Hz

wl_max = 2.4 # um
fr_min = c/(wl_max*1e-6) #Hz

wl_num = 41

fr_p = np.linspace(fr_max, fr_min, wl_num) # 1/s
wl_p = c/fr_p * 1e6 # 1/s

#  Retrieve from file
modes_directory = "./modes_full_vec/"
filename="wavelength_n_effs.dat"
n_effs = np.loadtxt(modes_directory + filename, delimiter  = ",")

# Calculate beta
beta =  (2*np.pi*fr_p*n_effs.T/c).T[:, 1]

frequencies_THz = fr_p/1e12 # Frequency in THz
beta_um = beta/1e6 # Beta in 1/um

n_effs_fit = PchipInterpolator(frequencies_THz, n_effs[:, 1])
beta_fit = PchipInterpolator(frequencies_THz, beta_um)

wg_len = 1e4 # micrometers
z_step = 1 # micrometers
z_num = int(wg_len/z_step + 0.5)
z = np.linspace(0.0, wg_len, z_num)

A_p = np.zeros(z_num, dtype='complex')
A_sh = np.zeros(z_num, dtype='complex')
A_s = np.zeros(z_num, dtype='complex')
A_i = np.zeros(z_num, dtype='complex')

A_p[0] = 1e-2+0j
A_sh[0] = 1e-8+0j
A_s[0] = 1e-8+0j
A_i[0] = 1e-8+0j

# All frequencies in THz
fr_p = 127
fr_sh = 2*fr_p
fr_s = 165
fr_i = fr_sh - fr_s

n_p = n_effs_fit(fr_p)
n_sh = n_effs_fit(fr_sh)
n_s = n_effs_fit(fr_s)
n_i = n_effs_fit(fr_i)
print(n_i)
c = c*1e-6
Z_0 = Z_0*(1e6)**2 * (1e-12)**3
deff = deff*1e22

k_0 = 1 - z_step*alpha/2
k_p = deff*2*np.pi*fr_p*np.sqrt(2*Z_0)/(c*n_p*np.sqrt(n_sh))
k_shg = deff*2*np.pi*fr_sh*np.sqrt(Z_0)/(c*n_p*np.sqrt(2*n_sh))
k_dfg = deff*2*np.pi*fr_sh*np.sqrt(2*Z_0)/(c*np.sqrt(n_sh*n_s*n_i))
k_s = deff*2*np.pi*fr_s*np.sqrt(2*Z_0)/(c*np.sqrt(n_sh*n_s*n_i))
k_i = deff*2*np.pi*fr_i*np.sqrt(2*Z_0)/(c*np.sqrt(n_sh*n_s*n_i))
print(k_s)
print(k_i)

mismatch = 5 # Lambda in um


d_beta_shg = 2*beta_fit(fr_p) - beta_fit(fr_sh) + 2*np.pi/mismatch
d_beta_dfg = beta_fit(fr_s) + beta_fit(fr_i) - beta_fit(fr_sh) + 2*np.pi/mismatch

for i in range(1, z_num):
    A_p[i] = k_0*A_p[i-1] + 1j*z_step*k_p  * A_sh[i-1]*np.conj(A_p[i-1])*np.exp(-1j*d_beta_shg*z[i-1])
    A_sh[i] =k_0*A_sh[i-1]+ 1j*z_step*k_shg* A_p[i-1]*A_p[i-1] * np.exp(1j*d_beta_shg*z[i-1]) + 1j*z_step*k_dfg * A_s[i-1] * A_i[i-1] * np.exp(1j*d_beta_dfg*z[i-1])
    A_s[i] = k_0*A_s[i-1] + 1j*z_step*k_s  * A_sh[i-1]*np.conj(A_i[i-1])*np.exp(-1j*d_beta_dfg*z[i-1])
    A_i[i] = k_0*A_i[i-1] + 1j*z_step*k_i  * A_sh[i-1]*np.conj(A_s[i-1])*np.exp(-1j*d_beta_dfg*z[i-1])


plt.plot(z, np.abs(A_s)**2, label="$A_s$")
plt.xlabel("Z [um]")
plt.ylabel("$|A_k|^2$")
plt.legend()
plt.show()