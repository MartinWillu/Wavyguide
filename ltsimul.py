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

wl = 1.55 #Initial test wavelength
x_step = 0.01 # Simulation step x-direction
y_step = 0.01 # Simulation step y-direction
wg_height = 0.6 # Height of waveguide
wg_width = 1.5 # Width of waveguide top
sub_height = 1.0 # Height of substrate
sub_width = 4. # Width of substrate
clad_height = 1.0 # Height of cladding above top  of waveguide  
film_thickness = 0.7 # Thickness of film below waveguide
angle = 80. # Waveguide wall angle
cut = 'x' # Crystal axis normal to cut
mode_num = 4 # Number of modes to simulate

# Easily change n of structure
def struct_func(n_sub, n_wg, n_clad):
    return RidgeWaveguide(wl, x_step, y_step, wg_height, wg_width,
                             sub_height, sub_width, clad_height,
                             n_sub, n_wg, angle, n_clad, film_thickness)
fr_min = 2e14
fr_max = 1e15
fr_step = 2e13
fr_num = int((fr_max-fr_min)/fr_step)+1
frequencies = np.linspace(fr_min, fr_max, fr_num) 

wavelengths = c/frequencies *1e6



n_sub = mat.SiO2().n(wl) # Refractive index of the substrate
ltani = mat.LtAni(cut) # Refractive indices of the waveguide.
n3 = ltani.n3(wl)
n_wg_xx = n3[0]
n_wg_yy = n3[1]
n_wg_zz = n3[2]
n_clad = mat.SiO2().n(wl) # Refractive index of the cladding

# Make structure for fully vectorial solution
struct_xx = struct_func(n_sub, n_wg_xx, n_clad) 
struct_yy = struct_func(n_sub, n_wg_yy, n_clad)
struct_zz = struct_func(n_sub, n_wg_zz, n_clad)
struct_ani = StructureAni(struct_xx, struct_yy, struct_zz)
struct_ani.write_to_file() # To be combined with plot of mode


# Create solver
solver = ModeSolverFullyVectorial(mode_num, tol=1e-9, boundary="SS00")

# Only for wl = 1550 nm to plot modes
solver.solve(struct_ani) 
solver.write_modes_to_file() 

# solver.solve_ng(struct_ani, 0.01) # Group index at 1550 nm

# Calculate effective refractive indices for all wavelengths
# n_effs = solver.solve_sweep_wavelength(struct_ani, wavelengths) 


#  Retrieve from file
modes_directory = "./modes_full_vec/"
filename="wavelength_n_effs.dat"
n_effs = np.loadtxt(modes_directory + filename, delimiter  = ",")

# Calculate beta
beta =  (2*np.pi*frequencies*n_effs.T/c).T[:, 1]
beta_fit = Akima1DInterpolator(frequencies, beta, method='akima')

# derivatives too small

plt.clf()
plt.plot(frequencies, beta, 'o', label="simul")
plt.plot(frequencies, beta_fit(frequencies), '-', label="Fit")
plt.legend()
plt.title("Beta fit")
plt.ylabel("Beta [$m /s^2$]")
plt.xlabel("Frequency [1/s]")
plt.grid()
plt.savefig(modes_directory + "fit_plot.png")

plt.clf()
plt.plot(frequencies, beta_fit(frequencies, nu = 1))
plt.title("First derivative of beta")
plt.xlabel("Frequency [1/s]")
plt.ylabel(r'$\beta_1$')
plt.grid()
plt.savefig(modes_directory + "beta_1.png")

plt.clf()
plt.plot(frequencies, beta_fit(frequencies, nu = 2))
plt.title("Second derivative of beta")
plt.xlabel("Frequency [1/s]")
plt.ylabel(r'$\beta_2$')
plt.grid()
plt.savefig(modes_directory + "beta_2.png")

plt.clf()
plt.plot(frequencies, beta_fit(frequencies, nu = 3))
plt.title("Third derivative of beta")
plt.xlabel("Frequency [1/s]")
plt.ylabel(r'$\beta_3$')
plt.grid()
plt.savefig(modes_directory + "beta_3.png")

plt.clf()
plt.plot(frequencies, beta_fit(frequencies, nu = 4))
plt.title("Fourth derivative of beta")
plt.xlabel("Frequency [1/s]")
plt.ylabel(r'$\beta_4$')
plt.grid()
plt.savefig(modes_directory + "beta_4.png")