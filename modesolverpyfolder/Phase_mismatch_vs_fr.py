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


wl = 1.55 # Just a placeholder to make initial structure
x_step = 0.01 # Simulation step x-direction
y_step = 0.01 # Simulation step y-direction
wg_width = 1 
wg_height = 0.6
sub_height = 3 # Height of substrate
sub_width = 7 # Width of substrate
clad_height = 3 # Height of cladding above top  of waveguide  
film_thickness = 0.70 # Thickness of film including waveguide
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
def struct_func(n_sub, n_wg, n_clad):
    return RidgeWaveguide(wl, x_step, y_step, wg_height, wg_width,
                             sub_height, sub_width, clad_height,
                             n_sub, n_wg, angle, n_clad, film_thickness, half)

# SEND CALLABLE n TO STRUCTURE
n_sub = mat.SiO2().n # Refractive index of the substrate
lnani = mat.LnAni(cut) # Refractive indices of the waveguide with cut x. 
n_wg_xx = lnani.xx.n
n_wg_yy = lnani.yy.n
n_wg_zz = lnani.zz.n
n_clad = mat.SiO2().n # Refractive index of the cladding

# Make structure for fully vectorial solution
struct_xx = struct_func(n_sub, n_wg_xx, n_clad) 
struct_yy = struct_func(n_sub, n_wg_yy, n_clad)
struct_zz = struct_func(n_sub, n_wg_zz, n_clad)
struct_ani = StructureAni(struct_xx, struct_yy, struct_zz)
struct_ani.write_to_file()

wl_min = 0.8 # um
fr_max = c/(wl_min*1e-6) # Hz

wl_max = 2.4 # um
fr_min = c/(wl_max*1e-6) #Hz

wl_num = 41

fr_p = np.linspace(fr_max, fr_min, wl_num) # 1/s
wl_p = c/fr_p * 1e6 # 1/s

fr_sh = fr_p*2 # 1/s
wl_sh = c/fr_sh *1e6 # um

modes_directory = "./modes_full_vec/"
f_name = "neff_vs_wl_ln_1_um"
file_fractions = "fraction_te_"

# Took one whole night to run
solver.solve_sweep_wavelength(struct_ani, wl_p, filename = f_name + "_pump.dat")
solver.solve_sweep_wavelength(struct_ani, wl_sh, filename = f_name + "_shg.dat")

#  Retrieve from file
te_fractions_p = np.loadtxt(modes_directory + file_fractions + f_name + "_pump.dat", delimiter=",")[:, 1:]
te_fractions_sh = np.loadtxt(modes_directory + file_fractions + f_name + "_shg.dat", delimiter=",")[:, 1:]
n_effs_p = np.loadtxt(modes_directory + f_name + "_pump.dat", delimiter  = ",")[:, 1:]
n_effs_sh = np.loadtxt(modes_directory + f_name + "_shg.dat", delimiter  = ",")[:, 1:]


plt.clf()
plt.scatter((wl_p*np.ones((mode_num, wl_num))).T, n_effs_p, linewidths=0.01, c=te_fractions_p, cmap='viridis')
plt.scatter((wl_sh*np.ones((mode_num, wl_num))).T, n_effs_sh, linewidths=0.01, c=te_fractions_sh, cmap='viridis')
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

"""
Add code to remove spikes due to hybrid modes from plot
Maybe: If  valid_modes_p[1][unique_p[1]] or valid_modes_sh[1][unique_sh[1]] moves up instead of down, skip

"""

beta_p = first_te_p*2*np.pi*fr_p[unique_p[0]]/c # 1/m
beta_sh = first_te_sh*2*np.pi*fr_sh[unique_sh[0]]/c # 1/m

d_beta = np.abs(2*beta_p - beta_sh)

poling_period = 2*np.pi/d_beta

plt.clf()
plt.plot(wl_p, poling_period*1e6)
plt.xlabel("Pump Wavelength [$\mu m$]")
plt.ylabel(r'Poling period [$\mu m$]')
plt.title("Poling period vs wavelength")
plt.grid()
plt.savefig("poling_vs_wl_ln_te_test.png")

""" Must be updated
frequencies_THz = fr_p/1e12 #Frequency in THz
beta_mm = beta_p/1e3 # Beta in 1/mm

beta_fit = Akima1DInterpolator(frequencies_THz, beta_mm, method='akima')
n_eff_fit = Akima1DInterpolator(frequencies_THz, n_effs_p, method='akima')

dfrf = np.asarray(np.gradient(n_eff_fit(frequencies_THz), frequencies_THz, axis=0))


plt.clf()
plt.plot(frequencies_THz, n_eff_fit(frequencies_THz, nu = 1), '-', label  ="Interp")
plt.plot(frequencies_THz, dfrf, '--', label="Gradient")
plt.legend()
plt.title("$n_{eff}$ first derivative")
plt.ylabel("n [1]")
plt.xlabel("Frequency [THz]")
plt.legend(["$TE_{00, interp}$", "$TM_{00, interp}$", "$TE_{10, gradient}$", "$TM_{10, gradient}$"])
plt.grid()
plt.savefig(modes_directory + "n_eff_d.png")

plt.clf()
plt.plot(frequencies_THz, beta_fit(frequencies_THz), '-', label  ="Plot")
plt.legend()
plt.title("Beta fit")
plt.ylabel("Beta [1/mm]")
plt.xlabel("Frequency [THz]")
plt.legend(["$TE_{00}$", "$TM_{00}$"])
plt.grid()
plt.savefig(modes_directory + "beta.png")

first_derivative = np.asarray(np.gradient(beta_fit(frequencies_THz), frequencies_THz, axis=0))/(2*np.pi)

chain_rule_betad = (n_eff_fit(frequencies_THz) + (frequencies_THz*dfrf.T).T)/c * 1e9


plt.clf()
plt.plot(frequencies_THz, first_derivative, label="Plot")
plt.plot(frequencies_THz, chain_rule_betad, '--', label="Chain")
plt.plot(frequencies_THz, beta_fit(frequencies_THz, nu=1)/(2*np.pi), '-x')
plt.title("First derivative of beta")
plt.xlabel("Frequency [THz]")
plt.ylabel(r'$\beta_1$ [$mm^{-1} rad^{-1} THz^{-1}$]')
plt.grid()
plt.legend(["$TE_{00, gradient}$", "$TM_{00, gradient}$", "$TE_{00, chain}$", 
            "$TM_{00, chain}$", "$TE_{00, interp}$", "$TM_{00, interp}$"])
plt.savefig(modes_directory + "beta_1.png")
second_derivative = np.asarray(np.gradient(first_derivative, frequencies_THz, axis=0))/(2*np.pi)

plt.clf()
plt.plot(frequencies_THz[2:-3], second_derivative[2:-3, :mode_num+1], '-', label="Plot")
plt.plot(frequencies_THz[2:-3], beta_fit(frequencies_THz, nu=2)[2:-3, :mode_num+1]/(2*np.pi)**2, '--', label="Something")
plt.title("Second derivative of beta")
plt.xlabel("Frequency [THz]")
plt.ylabel(r'$\beta_2$ [$mm^{-2} rad^{-2} THz^{-2}$]')
plt.grid()
plt.legend(["$TE_{00, gradient}$", "$TM_{00, gradient}$", "$TE_{00, interp}$", "$TM_{00, interp}$"])
plt.savefig(modes_directory + "beta_2.png")

third_derivative = np.asarray(np.gradient(second_derivative, frequencies_THz, axis=0))/(2*np.pi)
plt.clf()
plt.plot(frequencies_THz[3:-4], third_derivative[3:-4, :mode_num+1], label="Plot")
plt.title("Third derivative of beta")
plt.xlabel("Frequency THz")
plt.ylabel(r'$\beta_3$ [$mm^{-3} rad^{-3} THz^{-3}$]')
plt.grid()
plt.legend(["$TE_{00, gradient}$", "$TM_{00, gradient}$"])
plt.savefig(modes_directory + "beta_3.png")

fourth_derivative = np.asarray(np.gradient(third_derivative, frequencies_THz, axis=0))/(2*np.pi)
plt.clf()
plt.plot(frequencies_THz[4:-5], fourth_derivative[4:-5, :mode_num+1], label="Plot")
plt.title("Fourth derivative of beta")
plt.xlabel("Frequency [THz]")
plt.ylabel(r'$\beta_4 [$mm^{-4} rad^{-4} THz^{-4}$]$')
plt.grid()
plt.legend(["$TE_{00, gradient}$", "$TM_{00, gradient}$"])
plt.savefig(modes_directory + "beta_4.png") """



