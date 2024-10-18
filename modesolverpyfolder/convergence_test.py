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
def struct_func(wl, n_sub, n_wg, n_clad, step):
    return RidgeWaveguide(wl, step, step, wg_height, wg_width,
                             sub_height, sub_width, clad_height,
                             n_sub, n_wg, angle, n_clad, film_thickness, half)

def shg_struct_func(wl_p, step):
    wl_sh = wl_p/2
    n_sub = mat.SiO2().n(wl_p) # Refractive index of the substrate
    lnani = mat.LnAni(cut) # Refractive indices of the waveguide with cut x.
    n3 = lnani.n3(wl_p) 
    n_wg_xx = n3[0]
    n_wg_yy = n3[1]
    n_wg_zz = n3[2]
    n_clad = mat.Air().n(wl_p) # Refractive index of the cladding

    # Make structure for fully vectorial solution
    struct_xx = struct_func(wl_p, n_sub, n_wg_xx, n_clad, step) 
    struct_yy = struct_func(wl_p, n_sub, n_wg_yy, n_clad, step)
    struct_zz = struct_func(wl_p, n_sub, n_wg_zz, n_clad, step)
    struct_ani_p = StructureAni(struct_xx, struct_yy, struct_zz)

    n_sub = mat.SiO2().n(wl_sh) # Refractive index of the substrate
    lnani = mat.LnAni(cut) # Refractive indices of the waveguide with cut x.
    n3 = lnani.n3(wl_sh) 
    n_wg_xx = n3[0]
    n_wg_yy = n3[1]
    n_wg_zz = n3[2]
    n_clad = mat.Air().n(wl_sh) # Refractive index of the cladding

    # Make structure for fully vectorial solution
    struct_xx = struct_func(wl_sh, n_sub, n_wg_xx, n_clad, step) 
    struct_yy = struct_func(wl_sh, n_sub, n_wg_yy, n_clad, step)
    struct_zz = struct_func(wl_sh, n_sub, n_wg_zz, n_clad, step)
    struct_ani_sh = StructureAni(struct_xx, struct_yy, struct_zz)
    return struct_ani_p, struct_ani_sh


wl_p = 1.55  # um
fr_p = c/wl_p*1e6 # 1/s

wl_sh = 0.775
fr_sh = c/wl_sh*1e6

structures = np.array([])
steps = np.arange(0.010, 0.006, -0.0002)
param  =np.array([])
for s in steps:
    structure_p, structure_sh = shg_struct_func(wl_p, s)
    structures = np.hstack([structures, structure_p, structure_sh])
    param = np.hstack([param, s, s])

structures[0].write_to_file()

# solver.solve_sweep_structure(structures, param, plot=False)

#  Retrieve from file
modes_directory = "./modes_full_vec/"
filename="structure_n_effs.dat"
txt_load = np.loadtxt(modes_directory + filename, delimiter  = ",")

n_effs_p = txt_load[::2, 1]
n_effs_sh = txt_load[1::2, 2]

beta_p = n_effs_p*2*np.pi*fr_p/c # 1/m
beta_sh = n_effs_sh*2*np.pi*fr_sh/c # 1/m

d_beta = np.abs(2*beta_p - beta_sh)

poling_period = 2*np.pi/d_beta

plt.clf()
plt.plot(1/steps, poling_period*1e6)
plt.xlabel("Inverse step size [$1/\mu m$]")
plt.ylabel(r'Poling period [$\mu m$]')
plt.title("Phase mismatch vs step size")
plt.grid()
plt.savefig("convergence_test.png")



