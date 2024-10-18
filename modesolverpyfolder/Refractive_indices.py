import sys
import os
sys.path.append("C:\\Users\\kmari\\OneDrive\\Documents\\Thesis\\Wavyguide-main\\modesolverpyfolder")

import numpy as np
import opticalmaterialspy as mat
from modesolverpy.structure import RidgeWaveguide
from modesolverpy.mode_solver import ModeSolverFullyVectorial
from modesolverpy.structure_base import StructureAni
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator

c = 2.998e8
rng = np.random.default_rng()
wavelengths = np.linspace(0.3, 3, 101)
wl = 1.55
cut = "x"

n_sub = mat.SiO2().n # Refractive index of the substrate
ltani = mat.LnAni(cut) # Refractive indices of the waveguide.
n3 = ltani.n3
n_wg_xx = n3[0]
n_wg_yy = n3[1]
n_wg_zz = n3[2]
n_clad = mat.SiO2().n # Refractive index of the cladding

wavelengths_less = np.linspace(0.3, 3, 50)

modes_directory = "./modes_full_vec/"

plt.clf()
plt.plot(wavelengths, n_sub(wavelengths), 'r-', label="$SiO_2$")
plt.plot(wavelengths, n_wg_xx(wavelengths), 'g-', label="$LN_x$")
plt.plot(wavelengths, n_wg_yy(wavelengths), 'y-', label="$LN_y$")
plt.plot(wavelengths_less, n_wg_zz(wavelengths), 'bx', label="$LN_z$")
plt.legend()
plt.title("Bulk refractive indices")
plt.ylabel("n")
plt.xlabel("Wavelength [$\mu m$]")
plt.grid() 
plt.savefig(modes_directory + "refractive_indices_LN.png")