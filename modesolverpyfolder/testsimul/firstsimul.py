import numpy as np
import opticalmaterialspy as mat
from modesolverpy.structure import RidgeWaveguide
from modesolverpy.mode_solver import ModeSolverFullyVectorial
from modesolverpy.structure_base import StructureAni
import matplotlib as plt


wl = 1.55
x_step = 0.06
y_step = 0.06
wg_height = 0.8
wg_width = 1.8
sub_height = 1.0
sub_width = 4.
clad_height = 1.0
film_thickness = 1.2
angle = 77.

def struct_func(n_sub, n_wg, n_clad):
    return RidgeWaveguide(wl, x_step, y_step, wg_height, wg_width,
                             sub_height, sub_width, clad_height,
                             n_sub, n_wg, angle, n_clad, film_thickness)

n_sub = mat.SiO2().n(wl)
n_wg_xx = mat.Ktp('x').n(wl)
n_wg_yy = mat.Ktp('y').n(wl)
n_wg_zz = mat.Ktp('z').n(wl)
n_clad = mat.Air().n(wl)

struct_xx = struct_func(n_sub, n_wg_xx, n_clad)
struct_yy = struct_func(n_sub, n_wg_yy, n_clad)
struct_zz = struct_func(n_sub, n_wg_zz, n_clad)

struct_ani = StructureAni(struct_xx, struct_yy, struct_zz)
struct_ani.write_to_file()

solver = ModeSolverFullyVectorial(8)
#solver.solve(struct_ani)
#solver.write_modes_to_file()

#solver.solve_ng(struct_ani, 0.01)

#solver.solve_sweep_wavelength(struct_ani, np.linspace(1.501, 1.60, 21))



print(struct_ani.n)

print(np.unique(struct_ani.n))