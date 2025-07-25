"""Single-mode Rayleigh-Taylor

Navier-Stokes equations with Boussinesq buoyancy terms set up for a
single mode Rayleigh-Taylor simulation.  The initial disturbance is a
velocity disturbance that roughly corresponds to the linear
instability eigenfunction.
"""
import numpy
from psdns import *
from psdns.equations.navier_stokes import Boussinesq
from scipy.integrate import quad

grid = SpectralGrid(
    sdims=[21, 21, 43],#[11, 11, 43],
    pdims=[32, 32, 64],#[16, 16, 64],
    box_size=[4*numpy.pi, 4*numpy.pi, 8*numpy.pi]
    )
folder = "test_IC/"
grid.checkpoint(folder + "data.grid")
equations = Boussinesq(Re=10)#000)#(Re=1)

top_hat_spectrum = lambda k, kmin, kmax: numpy.ones(k.shape)*5e-2
inverse_cubic_spectrum = lambda k, kmin, kmax: (k-(kmin-1))**(-3)*5e-2
negative_inverse_cubic_spectrum = lambda k, kmin, kmax: (-k+(kmax+1))**(-3)*5e-2
k_c = 2
xixi = 0.25
n = 2
a = n
R_C = lambda k: (k/k_c)**n * numpy.exp(-0.5*a*(k/k_c)**2)
R_C_spectrum = lambda k, kmin, kmax: numpy.sqrt(xixi * R_C(k)/quad(R_C, 0, numpy.inf)[0])

x = grid.x[:2,:,:,0]
solver = RungeKutta(
    dt=0.01,
    tfinal=20,
    equations=equations,
    ic=equations.perturbed_interface(grid, equations.phys_space_IC(grid, 2, 12, top_hat_spectrum), 0.1, 0.1),
    diagnostics=[
        FieldDump(tdump=1, grid=grid, filename=folder + "data{:04g}"),
        VTKDump(
        #VTKDumpParallel(
            tdump=1, grid=grid,
            filename=folder + "./phys{time:04g}",
            names=['U', 'V', 'W', 'C' ]
            )
        ],
    )
solver.run()
solver.print_statistics()