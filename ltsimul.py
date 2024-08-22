import sys
sys.path.append("C:\\Users\\kmari\\OneDrive\\Documents\\Thesis\\Wavyguide-main\\modesolverpyfolder")

import numpy as np
import opticalmaterialspy as mat
from modesolverpy.structure import RidgeWaveguide
from modesolverpy.mode_solver import ModeSolverFullyVectorial
from modesolverpy.structure_base import StructureAni
import matplotlib as plt


wl = 1.55
x_step = 0.02
y_step = 0.02
wg_height = 0.7
wg_width = 1.0
sub_height = 2.0
sub_width = 4.
clad_height = 1.0
film_thickness = 1.2
angle = 77.
cut = 'x'

def struct_func(n_sub, n_wg, n_clad):
    return RidgeWaveguide(wl, x_step, y_step, wg_height, wg_width,
                             sub_height, sub_width, clad_height,
                             n_sub, n_wg, angle, n_clad, film_thickness)

n_sub = mat.SiO2().n(wl)
ltani = mat.LtAni(cut)
n3 = ltani.n3(wl)
n_wg_xx = n3[0]
n_wg_yy = n3[1]
n_wg_zz = n3[2]
n_clad = mat.SiO2().n(wl)

struct_xx = struct_func(n_sub, n_wg_xx, n_clad)
struct_yy = struct_func(n_sub, n_wg_yy, n_clad)
struct_zz = struct_func(n_sub, n_wg_zz, n_clad)

struct_ani = StructureAni(struct_xx, struct_yy, struct_zz)
struct_ani.write_to_file()

solver = ModeSolverFullyVectorial(4)
# solver.solve(struct_ani)
# solver.write_modes_to_file()

# solver.solve_ng(struct_ani, 0.01)

solver.solve_sweep_wavelength(struct_ani, np.linspace(1.501, 1.60, 31))