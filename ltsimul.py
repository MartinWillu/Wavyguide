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

wl_min = 1.785 # Start wavelength
wl_max = 2.280 # End wavelength
wl_d = 0.005 # Wavelength step
wl_num = int((wl_max-wl_min)/wl_d)+1 #Number of wavelengths to sweep over
wavelengths = np.linspace(wl_min, wl_max, wl_num) # Wavelengths to sweep over



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
solver = ModeSolverFullyVectorial(mode_num, tol=0.0)

""" # Only for wl = 1550 nm to plot modes
solver.solve(struct_ani) 
solver.write_modes_to_file() """

# solver.solve_ng(struct_ani, 0.01) # Group index at 1550 nm

# Calculate effective refractive indices for all wavelengths
n_effs = solver.solve_sweep_wavelength(struct_ani, wavelengths) 


#  Retrieve from file
modes_directory = "./modes_full_vec/"
filename="wavelength_n_effs.dat"
n_effs = np.loadtxt(modes_directory + filename, delimiter  = ",")

# Calculate beta
beta =  (n_effs[:, 1:].T/wavelengths).T*2*np.pi
diff  = np.diff(beta, n = 2, axis = 0)
beta2 = diff/(wl_d**2)

plt.clf()
plt.plot(wavelengths[1:-1], beta2, '-')
plt.grid()
legend_n = np.shape(beta2)[1]
legend = ["" for i in range(legend_n)]
for i in range(legend_n):
    legend[i] = "Mode " + str(i)

plt.legend(legend)
plt.title("Dispersion")
plt.ylabel("Dispersion")
plt.xlabel("Wavlength $\mu m$")
plt.savefig(modes_directory + "dispersion_curve.png")
# diff(x) --> out1[i] = x[i+1] - x[i]
# diff(x, n=2) --> out2[i] = out1[i+1]-out1[i] = x[i+2] - x[i+1] - (x[i+1] - x[i]) = x[i+2] - 2 x[i+1] + x[i]