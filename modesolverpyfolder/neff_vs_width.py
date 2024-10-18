import time
import numpy as np
import opticalmaterialspy as mat
from modesolverpy.structure import *
from modesolverpy.mode_solver import ModeSolverFullyVectorial
from modesolverpy.structure_base import StructureAni
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator

c = 2.998e8
rng = np.random.default_rng()

x_step = 0.01 # Simulation step x-direction
y_step = 0.01 # Simulation step y-direction
wg_width = 1.8
wg_height = 0.6
sub_height = 1.65 # Height of substrate
sub_width = 4. # Width of substrate
clad_height = 1.65 # Height of cladding above top  of waveguide  
film_thickness = 0.7 # Thickness of film including waveguide
angle = 80. # Waveguide wall angle
cut = 'x' # Crystal axis normal to cut
mode_num = 6 # Number of modes to simulate
half = True
if half:
    sub_width = sub_width/2
    # Create solver
    solver = ModeSolverFullyVectorial(mode_num, tol=1e-9, boundary="0S00")
else:
    # Create solver
    solver = ModeSolverFullyVectorial(mode_num, tol=1e-9, boundary="0000")


# Easily change a few parameters
def struct_func(wl, n_sub, n_wg, n_clad, wg_width):
    return RidgeWaveguide(wl, x_step, y_step, wg_height, wg_width,
                             sub_height, sub_width, clad_height,
                             n_sub, n_wg, angle, n_clad, film_thickness, half)

wl_p = 1.55  # um
fr_p = c/wl_p*1e6 # 1/s

wl_sh = wl_p/2
fr_sh = c/wl_sh*1e6

def shg_struct_func(film_thickness):
    n_sub = mat.SiO2().n # Refractive index of the substrate
    lnani = mat.LnAni(cut) # Refractive indices of the waveguide with cut x.
    n_wg_xx = lnani.xx.n
    n_wg_yy = lnani.yy.n
    n_wg_zz = lnani.zz.n
    n_clad = mat.Air().n # Refractive index of the cladding

    # Make structure for fully vectorial solution
    struct_xx = struct_func(wl_p, n_sub, n_wg_xx, n_clad, film_thickness) 
    struct_yy = struct_func(wl_p, n_sub, n_wg_yy, n_clad, film_thickness)
    struct_zz = struct_func(wl_p, n_sub, n_wg_zz, n_clad, film_thickness)
    struct_ani_p = StructureAni(struct_xx, struct_yy, struct_zz)

    # Make structure for fully vectorial solution
    struct_xx = struct_func(wl_sh, n_sub, n_wg_xx, n_clad, film_thickness) 
    struct_yy = struct_func(wl_sh, n_sub, n_wg_yy, n_clad, film_thickness)
    struct_zz = struct_func(wl_sh, n_sub, n_wg_zz, n_clad, film_thickness)
    struct_ani_sh = StructureAni(struct_xx, struct_yy, struct_zz)
    return struct_ani_p, struct_ani_sh
width_min = 1
width_max = 2
width_num = 51
widths = np.linspace(width_min, width_max, width_num)

structures = np.array([])
param  =np.array([])
for w in widths:
    structure_p, structure_sh = shg_struct_func(w)
    structures = np.hstack([structures, structure_p, structure_sh])
    param = np.hstack([param, w, w])

structures[-1].write_to_file()

#  Retrieve from file
modes_directory = "./modes_full_vec/"
f_name="structure_n_effs"
file_fractions = "te_fraction.dat"

solver.solve_sweep_structure(structures, param, filename=f_name, plot=False)

#  Retrieve from file
te_fractions_p = np.loadtxt(modes_directory + f_name + file_fractions, delimiter=",")[::2, 1:]
te_fractions_sh = np.loadtxt(modes_directory + f_name + file_fractions, delimiter=",")[1::2, 1:]
n_effs_p = np.loadtxt(modes_directory + f_name + "_pump" + ".dat", delimiter  = ",")[::2, 1:]
n_effs_sh = np.loadtxt(modes_directory + f_name + "_shg" + ".dat", delimiter  = ",")[1::2, 1:]

plt.clf()
plt.scatter((wl_p*np.ones((mode_num, width_num))).T, n_effs_p, linewidths=0.01, c=te_fractions_p, cmap='viridis')
plt.scatter((wl_sh*np.ones((mode_num, width_num))).T, n_effs_sh, linewidths=0.01, c=te_fractions_sh, cmap='viridis')
plt.xlabel("Wavelength ($\mu m$)")
plt.ylabel("$n_{eff}$")
plt.title("Effective index as a function of wavelength")
plt.grid()
plt.savefig("n_eff_vs_wl_ln_te_test")

# Find indices of only modes with te fraction higher than 90%
valid_modes_p = np.where(te_fractions_p>0.9)
valid_modes_sh = np.where(te_fractions_sh>0.9)

# Only one value per wavelength
unique_p = np.unique(valid_modes_p[0], return_index=True)
unique_sh = np.unique(valid_modes_sh[0], return_index=True)

# Make array of only first TE modes
first_te_p = n_effs_p[unique_p[0], valid_modes_p[1][unique_p[1]]]
first_te_sh = n_effs_sh[unique_sh[0], valid_modes_sh[1][unique_sh[1]]]

beta_p = first_te_p*2*np.pi*fr_p/c # 1/m
beta_sh = first_te_sh*2*np.pi*fr_sh/c # 1/m

d_beta = np.abs(2*beta_p - beta_sh)

poling_period = 2*np.pi/d_beta # m

plt.clf()
plt.plot(widths, poling_period*1e6)
plt.xlabel("Width [$\mu m$]")
plt.ylabel(r'Poling period [$\mu m$]')
plt.title("Phase mismatch vs Width")
plt.grid()
plt.savefig("width_varied_LN.png")





