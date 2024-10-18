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
wg_height = 0.35
sub_height = 1.65 # Height of substrate
sub_width = 4. # Width of substrate
clad_height = 1.65 # Height of cladding above top  of waveguide  
film_thickness = 0.6 # Thickness of film including waveguide
angle = 63. # Waveguide wall angle
cut = 'x' # Crystal axis normal to cut
mode_num = 2 # Number of modes to simulate
half = True
if half:
    sub_width = sub_width/2
    # Create solver
    solver = ModeSolverFullyVectorial(mode_num, tol=1e-9, boundary="0S00")
else:
    # Create solver
    solver = ModeSolverFullyVectorial(mode_num, tol=1e-9, boundary="0000")


# Easily change a few parameters
def struct_func(wl, n_sub, n_wg, n_clad, film_thickness):
    return RidgeWaveguide(wl, x_step, y_step, wg_height, wg_width,
                             sub_height, sub_width, clad_height,
                             n_sub, n_wg, angle, n_clad, film_thickness, half)

def shg_struct_func(wl_p, film_thickness):
    wl_sh = wl_p/2
    n_sub = mat.SiO2().n(wl_p) # Refractive index of the substrate
    ltani = mat.LnAni(cut) # Refractive indices of the waveguide with cut x.
    n3 = ltani.n3(wl_p) 
    n_wg_xx = n3[0]
    n_wg_yy = n3[1]
    n_wg_zz = n3[2]
    n_clad = mat.Air().n(wl_p) # Refractive index of the cladding

    # Make structure for fully vectorial solution
    struct_xx = struct_func(wl_p, n_sub, n_wg_xx, n_clad, film_thickness) 
    struct_yy = struct_func(wl_p, n_sub, n_wg_yy, n_clad, film_thickness)
    struct_zz = struct_func(wl_p, n_sub, n_wg_zz, n_clad, film_thickness)
    struct_ani_p = StructureAni(struct_xx, struct_yy, struct_zz)

    n_sub = mat.SiO2().n(wl_sh) # Refractive index of the substrate
    ltani = mat.LnAni(cut) # Refractive indices of the waveguide with cut x.
    n3 = ltani.n3(wl_sh) 
    n_wg_xx = n3[0]
    n_wg_yy = n3[1]
    n_wg_zz = n3[2]
    n_clad = mat.Air().n(wl_sh) # Refractive index of the cladding

    # Make structure for fully vectorial solution
    struct_xx = struct_func(wl_sh, n_sub, n_wg_xx, n_clad, film_thickness) 
    struct_yy = struct_func(wl_sh, n_sub, n_wg_yy, n_clad, film_thickness)
    struct_zz = struct_func(wl_sh, n_sub, n_wg_zz, n_clad, film_thickness)
    struct_ani_sh = StructureAni(struct_xx, struct_yy, struct_zz)
    return struct_ani_p, struct_ani_sh


wl_p = 1.55  # um
fr_p = c/wl_p*1e6 # 1/s

wl_sh = 0.775
fr_sh = c/wl_sh*1e6

structures = np.array([])
heights = np.linspace(0.57, 0.63, 7)
param  =np.array([])
for h in heights:
    structure_p, structure_sh = shg_struct_func(wl_p, h)
    structures = np.hstack([structures, structure_p, structure_sh])
    param = np.hstack([param, h, h])

structures[-1].write_to_file()

solver.solve_sweep_structure(structures, param, plot=False)

#  Retrieve from file
modes_directory = "./modes_full_vec/"
filename="structure_n_effs.dat"
txt_load = np.loadtxt(modes_directory + filename, delimiter  = ",")

n_effs_p = txt_load[::2, 1]
n_effs_sh = txt_load[1::2, 2]

beta_p = n_effs_p*2*np.pi*fr_p/c # 1/m
beta_sh = n_effs_sh*2*np.pi*fr_sh/c # 1/m


d_beta = beta_sh - 2*beta_p

poling_period = 2*np.pi/d_beta # m

plt.clf()
plt.plot(heights, poling_period*1e6)
plt.xlabel("Thickness [$\mu m$]")
plt.ylabel(r'Poling period [$\mu m$]')
plt.title("Phase mismatch vs thickness")
plt.grid()
plt.savefig("thickness_varied.png")





