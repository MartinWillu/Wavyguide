import sys
import os
sys.path.append("C:\\Users\\kmari\\OneDrive\\Documents\\Thesis\\Wavyguide-main\\modesolverpyfolder")

import numpy as np
import opticalmaterialspy as mat
from modesolverpy.structure import RidgeWaveguide
from modesolverpy.mode_solver import ModeSolverFullyVectorial
from modesolverpy.structure_base import StructureAni
import matplotlib.pyplot as plt


wl = 1.55 #Initial test wavelength
x_step = 0.005
y_step = 0.005
wg_height = 0.7
wg_width = 1.5
sub_height = 1.0
sub_width = 4.
clad_height = 1.0
film_thickness = 1.2
angle = 80.
cut = 'x'
mode_num = 4


wl_min = 0.28
wl_max = 1.6
wl_d = 0.01
wl_num = int((wl_max-wl_min)/wl_d)+1
wavelengths = np.linspace(wl_min, wl_max, wl_num) # Wavelengths to sweep over


#  Retrieve from file
modes_directory = "./modes_full_vec/"
filename="wavelength_n_effs.dat"
n_effs = np.loadtxt(modes_directory + filename, delimiter  = ",")

# Calculate beta
beta =  (n_effs[:, 1:].T*wavelengths).T*2*np.pi
diff  = np.diff(beta, n = 2, axis = 0)
beta2 = diff/(wl_d**2)

#Calculate phase mismatch
Lambda = 16
wl_min_i = int(wl_min/wl_d)
deltaBetaSHG = []
wl_SHG = []
for i in range(wl_min_i, wl_num):
    close = np.isclose(wavelengths, wavelengths[i]/2)
    if close.any():
        index_p = np.where(close)[0][0]
        deltaBeta = beta[index_p, :] - 2*beta[i, :] + Lambda
        deltaBetaSHG.append(deltaBeta)
        wl_SHG.append(wavelengths[i])
    

deltaBetaSHG = np.asarray(deltaBetaSHG).reshape((53, 4))
wl_SHG = np.asarray(wl_SHG)

plt.plot(wl_SHG, deltaBetaSHG)
plt.title("Second Harmoinc Generation phase mismatch, $\Lambda$ = 15 $\mu m$ ")
plt.ylabel("Mismatch")
plt.xlabel("Wavelength $\mu m$")
plt.grid()
plt.show()

# DFG
def DFG(w_p, w_s, L):
    w_shg = 2*w_p
    w_i = w_shg - w_s
    return w_i

# diff(x) --> out1[i] = x[i+1] - x[i]
# diff(x, n=2) --> out2[i] = out1[i+1]-out1[i] = x[i+2] - x[i+1] - (x[i+1] - x[i]) = x[i+2] - 2 x[i+1] + x[i]