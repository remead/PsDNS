"""Diagnostics to use with PsDNS solvers

In order to make it easier to obtain data from simulations, PsDNS
provides a standardized diagnostic interface.
:class:`~psdns.integrators.Intergrator` classes a provided with a user
specified list of diagnostics which are run after each time step.
This module contains a :class:`Diagnostic` base class, from which
users can create their own diagnostics, as well as several
library diagnostics to use.
"""
import csv
import sys
import warnings

from mpi4py import MPI

import numpy

from psdns import *


class Diagnostic(object):
    """Base class for diagnostics

    This is a base class which does not actually perform any
    diagnostic output, but contains some useful infrastructure.  The
    actual diagnostics are performed by overriding the
    :meth:`diagnostic` method.  The base class takes care of opening
    and closing a file to use for output, and triggering only at the
    correct output interval.

    In principle, any callable which takes the correct arguments can
    be used as a diagnostic, however, it is recommended that
    diagnostics be derived from this base class.
    """
    def __init__(self, tdump, grid, outfile=sys.stderr, append=False):
        """Create a diagnostic

        Each time the diagnostic is called, it checks to see if the
        simulation time has increased more than *tdump* since the last
        output, and if so, it triggers another diagnostic output.  The
        :class:`~psdns.bases.SpectralGrid` on which the data to be
        output is stored is given by *grid*.  *outfile* is either a
        file object to use for writing, or a filename which is opened
        for writing.
        """
        #: The dump interval
        self.tdump = tdump
        #: The simulation time when the most recent dump occured
        self.lastdump = -1e9
        self._needs_close = False
        if grid.comm.rank != 0:
            return
        self.append = append
        if hasattr(outfile, 'write'):
            #: The file in which to write output
            self.outfile = outfile
        else:
            self.outfile = open(outfile, 'a' if self.append else 'w')
            self._needs_close = True

    def __del__(self):
        if self._needs_close:
            self.outfile.close()

    def __call__(self, time, equations, uhat):
        if time - self.lastdump < self.tdump - 1e-8:
            return
        self.diagnostic(time, equations, uhat)
        self.lastdump = time

    def diagnostic(self, time, equations, uhat):
        """Empty diagnostic

        This is the method that actually computes whatever diagnostic
        quantities are desired and writes them to the output file.  In
        subclasses this should be over-ridden to perform whatever
        analysis is required.  The :meth:`diagnostics` method must
        take exactly three arguments, the current simulation *time*,
        an *equations* object representing the equations being solved,
        and a spectral data array, *uhat*, containing the PDE
        independent variables.
        """
        return NotImplemented


class StandardDiagnostics(Diagnostic):
    """Write statistical averages to a CSV file

    This is a flexible, extensible diagnostic routine for standard
    statistical properties of interesnt in Navier-Stokes simulations.
    It assumes that ``uhat[:3]`` will be the vector velocity field.
    """
    def __init__(self, fields=['tke', 'dissipation'], **kwargs):
        """Create a standard diagnostics object

        The *fields* argument is a list of names of methods, each one
        of which takes *equations* and *uhat* as arguments, and
        returns a scalar which is the value to write to the output.
        Users can sub-class :class:`StandardDiagnostics` to add any
        other scalar methods they need to compute.  The remainder of
        the arguments are the same as for the :class:`Diagnostics`
        class.
        """
        super().__init__(**kwargs)
        #: A list of names of methods to run corresponding to the
        #: columns of the output.
        self.fields = fields
        if kwargs['grid'].comm.rank != 0:
            return
        #: A Python standard library :class:`csv.DictWriter` for
        #: writing CSV output files.
        self.writer = csv.DictWriter(
            self.outfile,
            ['time'] + self.fields
            )
        if not self.append:
            self.writer.writeheader()

    def divU(self, equation, uhat):
        return uhat[:3].div().norm()

    def cmax(self, equations, uhat):
        cmax = uhat.grid.comm.reduce(
            numpy.amax(uhat[3].to_physical()),
            MPI.MAX,
            )
        if uhat.grid.comm.rank == 0:
            return cmax

    def cavg(self, equation, uhat):
        cavg = uhat[3].to_physical().average()
        if uhat.grid.comm.rank == 0:
            return cavg

    '''def growthrate(self, equation, uhat):
        cavg = uhat[3].to_physical().average()
        if uhat.grid.comm.rank == 0:
            c_light = -1
            c_heavy = 1
            X_st = 0.5
            X = (cavg - c_light)/(c_heavy - c_light)
            X_P = X/X_st if X <= X_st else (1-X)/(1-X_st)
            F1 = (cavg - c_light)/(c_heavy - c_light)
            F2 = (c_heavy - cavg)/(c_heavy - c_light)'''


    
    def tke(self, equations, uhat):
        """Compute the turbulent kinetic energy"""
        u2 = uhat[:3].norm()
        if uhat.grid.comm.rank == 0:
            return 0.5*u2

    def dissipation(self, equations, uhat):
        r"""Compute the dissipation rate of the turbulent kinetic energy

        The dissipation rate is given by

        .. math::
            :label:

            \varepsilon
            = 2 \nu
            \left<
              \frac{\partial u_i}{\partial x_j}
              \frac{\partial u_i}{\partial x_j}
            \right>
        """
        enstrophy = [(1j*uhat.grid.k[i]*uhat[j]).norm()
                     for i in range(3) for j in range(3)]
        if uhat.grid.comm.rank == 0:
            return equations.nu*sum(enstrophy)

    def urms(self, equations, uhat):
        """Compute <uu> velocity fluctuations"""
        urms = uhat[0].norm()
        if uhat.grid.comm.rank == 0:
            return urms

    def vrms(self, equations, uhat):
        """Compute <vv> velocity fluctuations"""
        vrms = uhat[1].norm()
        if uhat.grid.comm.rank == 0:
            return vrms

    def wrms(self, equations, uhat):
        """Compute <ww> velocity fluctuations"""
        wrms = uhat[2].norm()
        if uhat.grid.comm.rank == 0:
            return wrms

    def S(self, equations, uhat):
        r"""Compute the skewness

        The general tensor form for the skewness is

        .. math::
            :label: def-skewness

            S =
            \left<
                \frac{\partial u_i}{\partial x_k}
                \frac{\partial u_j}{\partial x_k}
                \frac{\partial u_i}{\partial x_j}
            \right>
        """
        gradu = uhat.grad().to_physical()
        S = [
            (gradu[i, k]*gradu[j, k]*gradu[i, j]).average()
            for i in range(3) for j in range(3) for k in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(S)

    def S2(self, equations, uhat):
        r"""Compute a generalized skewness

        Typically :math:`S` is generalized to higher-order moments of
        the first-derivative.  However, it can also be generalized in
        terms of higher-order derivatives, as

        .. math::
           :label:

           S_N =
           \left<
                \frac{\partial^N u_i}{\partial x_{j_1} \partial x_{j_2} \ldots \partial x_{j_N}}
                \frac{\partial^N u_j}{\partial x_{j_1} \partial x_{j_2} \ldots \partial x_{j_N}}
                \frac{\partial u_i}{\partial x_j}
           \right>

        Note that the regular skewness (eq. :eq:`def-skewness`) is
        given by :math:`S = S_1`.
        """
        gradu = uhat.grad()
        grad2u = gradu.grad().to_physical()
        gradu = gradu.to_physical()
        S = [
            (grad2u[i, k, l]*grad2u[j, k, l]*gradu[i, j]).average()
            for i in range(3) for j in range(3) for k in range(3)
            for l in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum([s.real for s in S])

    def S3(self, equations, uhat):
        gradu = uhat.grad()
        grad3u = gradu.grad().grad().to_physical()
        gradu = gradu.to_physical()
        S = [
            (grad3u[i, k, l, m]*grad3u[j, k, l, m]*gradu[i, j]).average()
            for i in range(3) for j in range(3) for k in range(3)
            for l in range(3) for m in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum([s.real for s in S])

    def S4(self, equations, uhat):
        gradu = uhat.grad()
        grad4u = gradu.grad().grad().grad().to_physical()
        gradu = gradu.to_physical()
        S = [
            (grad4u[i, k, l, m, n]*grad4u[j, k, l, m, n]*gradu[i, j]).average()
            for i in range(3) for j in range(3) for k in range(3)
            for l in range(3) for m in range(3) for n in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum([s.real for s in S])

    def G(self, equations, uhat):
        r"""Compute the palinstrophy

        We can define a generalized enstrophy,

        .. math::
            :label: palinstrophy

            G_N =
            \left<
                \frac{\partial^N u_i}{\partial x_{j_1} \partial x_{j_2} \ldots \partial x_{j_N}}
                \frac{\partial^N u_i}{\partial x_{j_1} \partial x_{j_2} \ldots \partial x_{j_N}}
            \right>

        The palinstrophy is :math:`G = G_2`.
        """
        G = [
            (-uhat.grid.k[j]*uhat.grid.k[l]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(G)

    def G3(self, equations, uhat):
        """Compute the :math:`G_3`

        Compute the generalized palinstrophy of the third order,
        :math:`G_3`, (see equation :eq:`palinstrophy`).
        """
        G = [
            (-1j*uhat.grid.k[j]*uhat.grid.k[l]*uhat.grid.k[m]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3) for m in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(G)

    def G4(self, equations, uhat):
        """Compute the :math:`G_4`

        Compute the generalized palinstrophy of the third order,
        :math:`G_4`, (see equation :eq:`palinstrophy`).
        """
        G = [
            (uhat.grid.k[j]*uhat.grid.k[l]*uhat.grid.k[m]*uhat.grid.k[n]
             *uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3) for m in range(3)
            for n in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(G)

    def G5(self, equations, uhat):
        """Compute the :math:`G_5`

        Compute the generalized palinstrophy of the third order,
        :math:`G_5`, (see equation :eq:`palinstrophy`).
        """
        G = [
            (1j*uhat.grid.k[j]*uhat.grid.k[l]*uhat.grid.k[m]*uhat.grid.k[n]
             *uhat.grid.k[o]*uhat[i]).norm()
            for i in range(3) for j in range(3)
            for l in range(3) for m in range(3)
            for n in range(3) for o in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(G)

    def H(self, equations, uhat):
        r"""Compute the H

        :math:`H` is a non-isotropic generalization of the :math:`h(r)`
        function in the von Karmen-Howarth equation.  It is defined by

        .. math::
            :label:

            H_n =
            \left<
                \frac{\partial^N u_i}{\partial x_{j_1} \partial x_{j_2} \ldots \partial x_{j_N}}
                \frac{\partial^{N+1} u_k u_i}{\partial x_k \partial x_{j_1} \partial x_{j_2} \ldots \partial x_{j_N}}
            \right>
        """
        uu = PhysicalArray(
            uhat.grid,
            numpy.einsum(
                "i...,j...->ij...",
                uhat.to_physical(),
                uhat.to_physical()
                )
            ).to_spectral()
        H = [
            ((1j*uhat.grid.k[p]*uhat[j]).to_physical()
             *(-uhat.grid.k[m]*uhat.grid.k[p]*uu[j,m]).to_physical()).average()
            for j in range(3) for m in range(3) for p in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(H)

    def H2(self, equations, uhat):
        uu = PhysicalArray(
            uhat.grid,
            numpy.einsum(
                "i...,j...->ij...",
                uhat.to_physical(),
                uhat.to_physical()
                )
            ).to_spectral()
        H = [
            ((-uhat.grid.k[p]*uhat.grid.k[q]*uhat[j]).to_physical()
             *(-1j*uhat.grid.k[m]*uhat.grid.k[p]*uhat.grid.k[q]*uu[j,m]).to_physical()).average()
            for j in range(3) for m in range(3) for p in range(3)
            for q in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(H)
        
    def H3(self, equations, uhat):
        uu = PhysicalArray(
            uhat.grid,
            numpy.einsum(
                "i...,j...->ij...",
                uhat.to_physical(),
                uhat.to_physical()
                )
            ).to_spectral()
        H = [
            ((-1j*uhat.grid.k[p]*uhat.grid.k[q]*uhat.grid.k[r]*uhat[j]).to_physical()
             *(uhat.grid.k[m]*uhat.grid.k[p]*uhat.grid.k[q]*uhat.grid.k[r]*uu[j,m]).to_physical()).average()
            for j in range(3) for m in range(3) for p in range(3)
            for q in range(3) for r in range(3)
            ]
        if uhat.grid.comm.rank == 0:
            return sum(H)
        
    def diagnostic(self, time, equations, uhat):
        """Write the diagnostics specified in :attr:`fields`"""
        row = dict(
            time=time,
            **{field: getattr(self, field)(equations, uhat)
               for field in self.fields}
            )
        if uhat.grid.comm.rank == 0:
            self.writer.writerow(row)
            self.outfile.flush()


class Profiles(Diagnostic):
    two_point_indicies = [
        (0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2),
        ]

    three_point_indicies = [
        (0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 0, 1), (0, 0, 2),
        (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 1), (0, 1, 2),
        ]

    header = "t = {}\nz u v w Rxx Ryy Rzz Rxy Rxz Ryz Rxxx Ryyy Rzzz Rxxy Rxxz Ryyx Ryyz Rzzx Rzzy Rxyz"
    
    def diagnostic(self, time, equations, uhat):
        ubar, u = uhat.to_physical().disturbance()
        Rij = [ (u[i]*u[j]).avg_xy() for i, j in self.two_point_indicies ]
        Rijk = [ (u[i]*u[j]*u[k]).avg_xy() for i, j, k in self.three_point_indicies ]
        if uhat.grid.comm.rank == 0:
            numpy.savetxt(
                self.outfile,
                (numpy.vstack([ uhat.grid.x[2,0,0,:], ubar ]
                              + Rij + Rijk )).T,
                header=self.header.format(time)
                )
            self.outfile.write("\n\n")
            self.outfile.flush()


class ProfilesWithConcentration(Profiles):
    two_point_indicies = [
        (0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2),
        (3, 3), (3, 0), (3, 1), (3, 2)
        ]

    three_point_indicies = [
        (0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 0, 1), (0, 0, 2),
        (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 1), (0, 1, 2),
        (3, 0, 0), (3, 1, 1), (3, 2, 2), (3, 0, 1), (3, 0, 2),
        (3, 1, 2), (3, 3, 0), (3, 3, 1), (3, 3, 2),
        ]

    header = "t = {}\nz u v w c Rxx Ryy Rzz Rxy Rxz Ryz cc ax ay az Rxxx Ryyy Rzzz Rxxy Rxxz Ryyx Ryyz Rzzx Rzzy Rxyz Rcxx Rcyy Rczz Rcxy Rcxz Rcyz Rccx Rccy Rccz"


class DissipationProfiles(Diagnostic):
    two_point_indicies = [ (0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2), ]
    
    header = "t = {}\nz epsxx epsyy epszz epsxy epsxz epsyz S G"
    
    def diagnostic(self, time, equations, uhat):
        uhat = uhat.disturbance()
        gradu = uhat.grad()
        grad2u = gradu.grad().to_physical()
        gradu = gradu.to_physical()
        epsij = [
            (gradu[i]*gradu[j]).avg_xy(axis=(0,))
            for i, j in self.two_point_indicies
            ]
        S =  [
            (gradu[i, k]*gradu[j, k]*gradu[i, j]).avg_xy()
            for i in range(3) for j in range(3) for k in range(3)
            ]
        G = (grad2u**2).avg_xy(axis=(0, 1, 2))
        if uhat.grid.comm.rank == 0:
            numpy.savetxt(
                self.outfile,
                (numpy.vstack([ uhat.grid.x[2,0,0,:], epsij, sum(S), G ])).T,
                header=self.header.format(time)
                )
            self.outfile.write("\n\n")
            self.outfile.flush()


class DissipationProfilesWithConcentration(DissipationProfiles):
    two_point_indicies = [
        (0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2),
        (3, 3), (3, 0), (3, 1), (3, 2)
        ]

    header = "t = {}\nz epsxx epsyy epszz epsxy epsxz epsyz epscc epscx epscy epscz S G"

    
class PressureProfiles(Diagnostic):
    press_routine = "pressure"

    num_components = 9

    header = "t = {}\nz p prms pu pv pw pc pdudx pdudy pdudz pdvdx pdvdy pdvdz pdwdx pdwdy pdwdz"

    def diagnostic(self, time, equations, uhat):
        ubar, u = uhat.to_physical().disturbance()
        pbar, p = getattr(equations, self.press_routine)(uhat).to_physical().disturbance()
        # Pressure diffusion flux
        pu = (p*u).avg_xy()
        pnorm = (p*p).avg_xy()
        # Pressure strain
        gradu = uhat.grad().to_physical()
        press_strain = (p*gradu).avg_xy()
        if uhat.grid.comm.rank == 0:
            numpy.savetxt(
                self.outfile,
                numpy.vstack([
                    uhat.grid.x[2,0,0,:], pbar, pnorm, pu,
                    press_strain.reshape((self.num_components, -1)) ]).T,
                header=self.header.format(time)
                )
            self.outfile.write("\n\n")
            self.outfile.flush()

class RapidPressProfiles(PressureProfiles):
    press_routine = "press_rapid"


class SlowPressProfiles(PressureProfiles):
    press_routine = "press_slow"


class BuoyantPressProfiles(PressureProfiles):
    press_routine = "press_buoyant"


class PressureProfilesWithConcentration(PressureProfiles):
    num_components = 12

    header = "t = {}\nz p prms pu pv pw pc pdudx pdudy pdudz pdvdx pdvdy pdvdz pdwdx pdwdy pdwdz pdcdx pdcdy pdcdz"


class RapidPressProfilesWithConcentration(PressureProfilesWithConcentration):
    press_routine = "press_rapid"


class SlowPressProfilesWithConcentration(PressureProfilesWithConcentration):
    press_routine = "press_slow"


class BuoyantPressProfilesWithConcentration(PressureProfilesWithConcentration):
    press_routine = "press_buoyant"


class Spectra(Diagnostic):
    r"""A diagnostic class for velocity spectra

    The full velocity spectrum is a tensor function of a
    three-dimensional wave-number,

    .. math::

        \Phi_{ij}(\boldsymbol{k}) 
        = \hat{u}_i(\boldsymbol{k}) \hat{u}_j^*(\boldsymbol{k})

    There are a number of different one-dimensional spectra that can
    be obtained from this quantity, which can all be related to each
    other in the simple case of isotropic turbulence.

    This diagnostic computes the one-dimensional energy spectrum
    function, defined as

    .. math::

        E(k)
        = \iint_{|\boldsymbol{k}|=k} \Phi_{ii}(\boldsymbol{k}) dS
    """
    def integrate_shell(self, u, dk, grid):
        nbins = int(grid.kmax/dk)+1
        spectrum = numpy.zeros([nbins])
        ispectrum = numpy.zeros([nbins], dtype=int)
        for k, v in numpy.nditer([grid.kmag, u]):
            spectrum[int(k/dk)] += v
            ispectrum[int(k/dk)] += 1
        k = numpy.arange(nbins)*dk
        spectrum = grid.comm.reduce(spectrum)
        ispectrum = grid.comm.reduce(ispectrum)
        if grid.comm.rank == 0:
            spectrum *= 4*numpy.pi*k**2/(ispectrum/3)/numpy.prod(grid.dk)
        return k, spectrum
        
    def diagnostic(self, time, equations, uhat):
        """Write the spectrum to a file

        The integral cannot be calculated as a simple Riemann sum,
        because the binning is not smooth enough.  Instead, this routine
        caclulates the shell average, and then multiplies by the shell
        surface area.
        """
        k, spectrum = self.integrate_shell(
            (uhat[:3]*uhat[:3].conjugate()).real/2,
            numpy.prod(uhat.grid.dk)**(1/3),
            uhat.grid
            )
        if uhat.grid.comm.rank == 0:
            for i, s in zip(k, spectrum):
                self.outfile.write("{} {}\n".format(i, s))
            self.outfile.write("\n\n")
            self.outfile.flush()


class Spectra2D_Parallel_concentration(Diagnostic):
    r"""A diagnostic class for concentration spectra in 2D
    """
    def __init__(self, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def integrate_shell(self, u, dk, grid):
        nbins = int(grid.kmax2D/dk)+1
        spectrum = numpy.zeros([nbins])
        ispectrum = numpy.zeros([nbins], dtype=int)
        midplane = grid.box_size[2]/2
        if midplane in grid.x[2,0,0,grid._local_z_slice.start:grid._local_z_slice.stop]:
            for k, v in numpy.nditer([grid.kmag_2D, u]):
                spectrum[int(k/dk)] += v
                ispectrum[int(k/dk)] += 1
        k = numpy.arange(nbins)*dk

        spectrum = grid.comm.reduce(spectrum, root=0)
        ispectrum = grid.comm.reduce(ispectrum, root=0)
        if grid.comm.rank == 0:
            spectrum *= 2*numpy.pi*k/(ispectrum/2)/numpy.prod(grid.dk[:2])*dk*self.scale
        return k, spectrum

    def diagnostic(self, time, equations, uhat):
        """Write the spectrum to a file

        The integral cannot be calculated as a simple Riemann sum,
        because the binning is not smooth enough.  Instead, this routine
        caclulates the shell average, and then multiplies by the shell
        surface area.
        """
        u = uhat.to_physical()
        uhat2d = u.to_spectral_2Dxy()
        midplane = uhat.grid.box_size[2]/2
        if midplane in uhat.grid.x[2,0,0,uhat.grid._local_z_slice.start:uhat.grid._local_z_slice.stop]:
            mid_idx = numpy.where(uhat.grid.x[2,0,0,uhat.grid._local_z_slice.start:uhat.grid._local_z_slice.stop] == midplane)[0][0]
            uhat2d = uhat2d[3,:,:, mid_idx]
        else:
            uhat2d = numpy.zeros(uhat2d[3,:,:, 0].shape)
        k, spectrum = self.integrate_shell(
            0.5 *  (uhat2d*uhat2d.conjugate()).real,
            numpy.prod(uhat.grid.dk[:2])**(1/2),
            uhat.grid)
        #if uhat.grid.comm.rank == 0:
            #print(spectrum)
        if uhat.grid.comm.rank == 0:
            for i, s in zip(k, spectrum):
                self.outfile.write("{} {}\n".format(i, s))
            self.outfile.write("\n\n")
            self.outfile.flush()


class Spectra2D_Parallel_velocity(Diagnostic):
    r"""A diagnostic class for concentration spectra in 2D
    """
    def __init__(self, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale


    def integrate_shell(self, u, dk, grid):
        nbins = int(grid.kmax2D/dk)+1
        spectrum = numpy.zeros([nbins])
        ispectrum = numpy.zeros([nbins], dtype=int)
        midplane = grid.box_size[2]/2
        if midplane in grid.x[2,0,0,grid._local_z_slice.start:grid._local_z_slice.stop]:
            for k, v in numpy.nditer([grid.kmag_2D, u]):
                spectrum[int(k/dk)] += v
                ispectrum[int(k/dk)] += 1
        k = numpy.arange(nbins)*dk

        spectrum = grid.comm.reduce(spectrum, root=0)
        ispectrum = grid.comm.reduce(ispectrum, root=0)
        if grid.comm.rank == 0:
            spectrum *= 2*numpy.pi*k/(ispectrum/2)/numpy.prod(grid.dk[:2])*dk*self.scale
        return k, spectrum

    def diagnostic(self, time, equations, uhat):
        """Write the spectrum to a file

        The integral cannot be calculated as a simple Riemann sum,
        because the binning is not smooth enough.  Instead, this routine
        caclulates the shell average, and then multiplies by the shell
        perimeter.
        """
        u = uhat.to_physical()
        uhat2d = u.to_spectral_2Dxy()
        midplane = uhat.grid.box_size[2]/2
        # Get data in the midplane
        if midplane in uhat.grid.x[2,0,0,uhat.grid._local_z_slice.start:uhat.grid._local_z_slice.stop]:
            mid_idx = numpy.where(uhat.grid.x[2,0,0,uhat.grid._local_z_slice.start:uhat.grid._local_z_slice.stop] == midplane)[0][0]
            uhat2d = uhat2d[:2,:,:, mid_idx]
        else:
            uhat2d = numpy.zeros(uhat2d[:2,:,:, 0].shape)
        k, spectrum = self.integrate_shell(
            0.5 *  (uhat2d*uhat2d.conjugate()).real,
            numpy.prod(uhat.grid.dk[:2])**(1/2),
            uhat.grid)

        if uhat.grid.comm.rank == 0:
            for i, s in zip(k, spectrum):
                self.outfile.write("{} {}\n".format(i, s))
            self.outfile.write("\n\n")
            self.outfile.flush()


class Spectra2D(Diagnostic):
    r"""A diagnostic class for concentration spectra in 2D
    """
    def __init__(self, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def integrate_shell(self, u, dk, grid):
        nbins = int(grid.kmax2D/dk)+1
        spectrum = numpy.zeros([nbins])
        ispectrum = numpy.zeros([nbins], dtype=int)
        for k, v in numpy.nditer([grid.kmag_2D, u]):
            spectrum[int(k/dk)] += v
            ispectrum[int(k/dk)] += 1
        k = numpy.arange(nbins)*dk
        spectrum *= 2*numpy.pi*k/(ispectrum/2)/numpy.prod(grid.dk[:2])*dk*self.scale
        #print(sum(spectrum*spectrum))
        #print(spectrum)
        #exit()
        return k, spectrum

    def diagnostic(self, time, equations, uhat):
        """Write the spectrum to a file

        The integral cannot be calculated as a simple Riemann sum,
        because the binning is not smooth enough.  Instead, this routine
        caclulates the shell average, and then multiplies by the shell
        surface area.
        """
        u = uhat.to_physical()
        u_plane = u[:, :, :, int(u.shape[3]/2)]
        uhat2d = numpy.fft.fft2(u_plane, axes=(-2, -1), norm="forward")
        N = uhat.grid.sdims
        i0 = numpy.array([*range(0, N[0]//2+1), *range(-((N[0]-1)//2), 0)])
        uhat2d = uhat2d[3, i0[:, None], i0]
        k, spectrum = self.integrate_shell(
            0.5 *  (uhat2d*uhat2d.conjugate()).real,
            numpy.prod(uhat.grid.dk[:2])**(1/2),
            uhat.grid)
        for i, s in zip(k, spectrum):
            self.outfile.write("{} {}\n".format(i, s))
        self.outfile.write("\n\n")
        self.outfile.flush()


class Integral_Length(Diagnostic):
    r"""A diagnostic class for concentration spectra in 2D
    """
    def __init__(self, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        

    def integrate_shell(self, u, dk, grid):
        nbins = int(grid.kmax2D/dk)+1
        spectrum = numpy.zeros([nbins])
        ispectrum = numpy.zeros([nbins], dtype=int)
        midplane = grid.box_size[2]/2
        if midplane in grid.x[2,0,0,grid._local_z_slice.start:grid._local_z_slice.stop]:
            for k, v in numpy.nditer([grid.kmag_2D, u]):
                spectrum[int(k/dk)] += v
                ispectrum[int(k/dk)] += 1
        k = numpy.arange(nbins)*dk

        spectrum = grid.comm.reduce(spectrum, root=0)
        ispectrum = grid.comm.reduce(ispectrum, root=0)
        if grid.comm.rank == 0:
            spectrum *= 2*numpy.pi*k/(ispectrum/2)/numpy.prod(grid.dk[:2])*dk*self.scale
        return k, spectrum
        
    def diagnostic(self, time, equations, uhat):
        """Write the spectrum to a file

        The integral cannot be calculated as a simple Riemann sum,
        because the binning is not smooth enough.  Instead, this routine
        caclulates the shell average, and then multiplies by the shell
        surface area.
        """
        u = uhat.to_physical()
        uhat2d = u.to_spectral_2Dxy()
        midplane = uhat.grid.box_size[2]/2
        if midplane in uhat.grid.x[2,0,0,uhat.grid._local_z_slice.start:uhat.grid._local_z_slice.stop]:
            mid_idx = numpy.where(uhat.grid.x[2,0,0,uhat.grid._local_z_slice.start:uhat.grid._local_z_slice.stop] == midplane)[0][0]
            uhat2d = uhat2d[:2,:,:, mid_idx]
        else:
            uhat2d = numpy.zeros(uhat2d[:2,:,:, 0].shape)
        k, spectrum = self.integrate_shell(
            0.5 *  (uhat2d*uhat2d.conjugate()).real, 
            numpy.prod(uhat.grid.dk[:2])**(1/2), 
            uhat.grid)
        
        if uhat.grid.comm.rank == 0:
            l_int = numpy.sum(spectrum[1:]/k[1:]) / numpy.sum(spectrum[1:])
            self.outfile.write("{} {}\n".format(time, l_int))
            self.outfile.flush()


class Integral_Length_noispectrum(Diagnostic):
    r"""A diagnostic class for concentration spectra in 2D
    """
    def __init__(self, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        

    def integrate_shell(self, u, dk, grid):
        nbins = int(grid.kmax2D/dk)+1
        spectrum = numpy.zeros([nbins])
        midplane = grid.box_size[2]/2
        if midplane in grid.x[2,0,0,grid._local_z_slice.start:grid._local_z_slice.stop]:
            for k, v in numpy.nditer([grid.kmag_2D, u]):
                spectrum[int(k/dk)] += v
        k = numpy.arange(nbins)*dk

        spectrum = grid.comm.reduce(spectrum)
        if grid.comm.rank == 0:
            spectrum *= 2*numpy.pi*k/numpy.prod(grid.dk[:2])*dk*self.scale
        return k, spectrum
        
    def diagnostic(self, time, equations, uhat):
        """Write the spectrum to a file

        The integral cannot be calculated as a simple Riemann sum,
        because the binning is not smooth enough.  Instead, this routine
        caclulates the shell average, and then multiplies by the shell
        surface area.
        """
        u = uhat.to_physical()
        uhat2d = u.to_spectral_2Dxy()
        midplane = uhat.grid.box_size[2]/2
        if midplane in uhat.grid.x[2,0,0,uhat.grid._local_z_slice.start:uhat.grid._local_z_slice.stop]:
            mid_idx = numpy.where(uhat.grid.x[2,0,0,uhat.grid._local_z_slice.start:uhat.grid._local_z_slice.stop] == midplane)[0][0]
            uhat2d = uhat2d[:2,:,:, mid_idx]
        else:
            uhat2d = numpy.zeros(uhat2d[:2,:,:, 0].shape)
        k, spectrum = self.integrate_shell(
            0.5 *  (uhat2d*uhat2d.conjugate()).real, 
            #0.5 *  numpy.sum((uhat2d*uhat2d.conjugate()).real, axis=0), 
            numpy.prod(uhat.grid.dk[:2])**(1/2), 
            uhat.grid)
        
        if uhat.grid.comm.rank == 0:
            l_int = numpy.sum(spectrum[1:]/k[1:]) / numpy.sum(spectrum[1:])
            self.outfile.write("{} {}\n".format(time, l_int))
            self.outfile.flush()


class Integral_Length_Cook(Diagnostic):
    r"""A diagnostic class for concentration spectra in 2D
    """
    def __init__(self, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def integrate_shell(self, u, dk, grid):
        nbins = int(grid.kmax2D/dk)+1
        spectrum = numpy.zeros([nbins])
        ispectrum = numpy.zeros([nbins], dtype=int)
        midplane = grid.box_size[2]/2
        if midplane in grid.x[2,0,0,grid._local_z_slice.start:grid._local_z_slice.stop]:
            for k, v in numpy.nditer([grid.kmag_2D, u]):
                spectrum[int(k/dk)] += v
                ispectrum[int(k/dk)] += 1
        k = numpy.arange(nbins)*dk

        spectrum = grid.comm.reduce(spectrum, root=0)
        ispectrum = grid.comm.reduce(ispectrum, root=0)
        if grid.comm.rank == 0:
            print(f"spectrum {spectrum}")
            print(f"ispectrum {ispectrum}")
            spectrum *= 2*numpy.pi*k/(ispectrum/2)/numpy.prod(grid.dk[:2])*dk*self.scale

            print(f"after spectrum {spectrum}")
            print(f"after spectrum divided by k {spectrum/k}")
            print(f"sum of after spectrum divided by k {numpy.sum(spectrum[1:]/k[1:])}")
            print(f"sum of after spectrum {numpy.sum(spectrum)}")
        return k, spectrum
        
    def diagnostic(self, time, equations, uhat):
        """Write the spectrum to a file

        The integral cannot be calculated as a simple Riemann sum,
        because the binning is not smooth enough.  Instead, this routine
        caclulates the shell average, and then multiplies by the shell
        surface area.
        """
        u = uhat.to_physical()
        u_ave = numpy.mean(u, axis=(1,2))
        #u = u-u_ave[:,None, None, :]
        uhat2d = u.to_spectral_2Dxy()
        midplane = uhat.grid.box_size[2]/2
        if midplane in uhat.grid.x[2,0,0,uhat.grid._local_z_slice.start:uhat.grid._local_z_slice.stop]:
            mid_idx = numpy.where(uhat.grid.x[2,0,0,uhat.grid._local_z_slice.start:uhat.grid._local_z_slice.stop] == midplane)[0][0]
            uhat2d = uhat2d[:2,:,:, mid_idx]
        else:
            uhat2d = numpy.zeros(uhat2d[:2,:,:, 0].shape)
        k, spectrum = self.integrate_shell(
            0.5 * (uhat2d*uhat2d.conjugate()).real, #numpy.sum((uhat2d*uhat2d.conjugate()).real, axis=0), 
            numpy.prod(uhat.grid.dk[:2])**(1/2), 
            uhat.grid)
        
        if uhat.grid.comm.rank == 0:
            #print(f"spectrum[0]:{spectrum[0]}")
            l_int = numpy.sum(spectrum[1:]/k[1:]) / numpy.sum(spectrum[1:])
            print(f"Time {time} l_int {l_int}\n\n")
            self.outfile.write("{} {}\n".format(time, l_int))
            self.outfile.flush()


class Integral_Length_Autocorrelate(Diagnostic):
    def correlate(self, u_fluc, axis, dx):
        from scipy.signal import correlate
        from scipy.integrate import trapezoid

        if axis == 0:
            nx, ny = u_fluc.shape
        elif axis == 1:
            ny, nx = u_fluc.shape
        else:
            print("Invalid axis. Exiting now, you're welcome")
            exit()
        R_sum = numpy.zeros(nx)
        for j in range(ny):
            if axis == 0:
                uj = u_fluc[:,j]
            else:
                uj = u_fluc[j,:]
            
            corr = correlate(uj, uj, mode = 'full')
            corr = corr[nx-1:]
            corr /= corr[0]

            R_sum += corr
        R = R_sum / ny

        r = numpy.arange(nx) * dx

        zero_idx = numpy.where(R <= 0)[0]

        if len(zero_idx) > 0:
            i_end = zero_idx[0]
        else:
            i_end = nx
        
        L = trapezoid(R[:i_end], r[:i_end])
        return L


    def diagnostic(self, time, equations, uhat):
        u = uhat.to_physical()
        midplane_loc = u.shape[3]//2
        midplane = u[:,:,:,u.shape[3]//2]
        u_vel = midplane[0]
        u_fluc = u_vel - numpy.mean(u_vel)
        v_vel = midplane[1]
        v_fluc = v_vel - numpy.mean(v_vel)

        u_long = self.correlate(u_fluc, 0, uhat.grid.dx[0])
        u_trans = self.correlate(u_fluc, 1, uhat.grid.dx[0])
        v_long = self.correlate(v_fluc, 0, uhat.grid.dx[1])
        v_trans = self.correlate(v_fluc, 1, uhat.grid.dx[1])

        print(f"Time: {time}, U longitudinal: {u_long}, U transveral: {u_trans}, V longitudinal: {v_long}, V transveral: {v_trans}")


class Integral_Length_SKE(Diagnostic):
    r"""A diagnostic class for concentration spectra in 2D
    """
    def __init__(self, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        

    def integrate_shell(self, u, dk, grid):
        nbins = int(grid.kmax2D/dk)+1
        spectrum = numpy.zeros([nbins])
        ispectrum = numpy.zeros([nbins], dtype=int)
        midplane = grid.box_size[2]/2
        if midplane in grid.x[2,0,0,grid._local_z_slice.start:grid._local_z_slice.stop]:
            for k, v in numpy.nditer([grid.kmag_2D, u]):
                spectrum[int(k/dk)] += v
                ispectrum[int(k/dk)] += 1
        k = numpy.arange(nbins)*dk

        spectrum = grid.comm.reduce(spectrum)
        #print(k[1:k.shape[0]])
        #print(spectrum[1:spectrum.shape[0]])
        k = k[1:k.shape[0]]
        spectrum = spectrum[1:spectrum.shape[0]]
        ispectrum = grid.comm.reduce(ispectrum)
        if grid.comm.rank == 0:
            #spectrum *= 2*numpy.pi*k/(ispectrum/2)/numpy.prod(grid.dk[:2])*dk*self.scale
            #spectrum /= ispectrum[1:ispectrum.shape[0]]
            l_int = 3*numpy.pi / 4 * numpy.sum(spectrum/k*dk)/numpy.sum(spectrum*dk)
            print(l_int)
            #print(numpy.sum(spectrum/k*dk))
            #print(numpy.sum(spectrum*dk))
            print("-_-_-_-_-_-_-_-_-_-_")

        return l_int #k, spectrum
        
    def diagnostic(self, time, equations, uhat):
        """Write the spectrum to a file

        The integral cannot be calculated as a simple Riemann sum,
        because the binning is not smooth enough.  Instead, this routine
        caclulates the shell average, and then multiplies by the shell
        surface area.
        """
        u = uhat.to_physical()
        uhat2d = u.to_spectral_2Dxy()
        midplane = uhat.grid.box_size[2]/2
        if midplane in uhat.grid.x[2,0,0,uhat.grid._local_z_slice.start:uhat.grid._local_z_slice.stop]:
            mid_idx = numpy.where(uhat.grid.x[2,0,0,uhat.grid._local_z_slice.start:uhat.grid._local_z_slice.stop] == midplane)[0][0]
            uhat2d = uhat2d[:2,:,:, mid_idx]
        else:
            uhat2d = numpy.zeros(uhat2d[:2,:,:, 0].shape)
        l_int = self.integrate_shell(
            0.5 *  numpy.sum((uhat2d*uhat2d.conjugate()).real, axis=0), 
            numpy.prod(uhat.grid.dk[:2])**(1/2), 
            uhat.grid)
        
        if uhat.grid.comm.rank == 0:
            #l_int = numpy.sum(spectrum[1:]/k[1:]) / numpy.sum(spectrum[1:])
            self.outfile.write("{} {}\n".format(time, l_int))
            self.outfile.flush()


class PlaneSpectra(Diagnostic):
    r"""A diagnostic class for 2D velocity spectra averaged across z-planes
    
    Averages the 2D energy spectrum over the midplane (z=0) and 5 planes
    above and below to capture the full vertical extent of the structures.
    """
    
    def integrate_shell_2d(self, energy, dk, grid):
        """Integrate energy over annular shells in 2D"""
        nbins = int(grid.kmax2D/dk) + 1
        spectrum = numpy.zeros([nbins])
        count = numpy.zeros([nbins], dtype=int)
        
        for kappa, e in numpy.nditer([grid.kmag_2D, energy], flags=['refs_ok']):
            if kappa > 0:
                idx = int(kappa/dk)
                if idx < nbins:
                    spectrum[idx] += e
                    count[idx] += 1
        
        k = numpy.arange(nbins) * dk
        spectrum = grid.comm.reduce(spectrum)
        count = grid.comm.reduce(count)
        
        if grid.comm.rank == 0:
            denom = count * dk**2
            denom[denom == 0] = 1
            spectrum = 2 * numpy.pi * k * spectrum / denom
        
        return k, spectrum
    
    def diagnostic(self, time, equations, uhat):
        """Write the averaged 2D spectrum to a file"""
        # Number of planes to sample above/below midplane
        n_planes_each_side = 2#5
        
        # Transform to 2D spectral
        u = uhat[:2].to_physical()
        uhat_2d = u.to_spectral_2Dxy()  # [2, kx, ky, nz_local]
        
        # Midplane index
        z_mid = uhat.grid.pdims[2] // 2
        
        # Accumulate energy across all sampled planes
        z_slice = uhat.grid._local_z_slice
        energy_sum = None
        n_contributing_planes = 0
        
        # Sample planes from z_mid-5 to z_mid+5
        for offset in range(-n_planes_each_side, n_planes_each_side + 1):
            z_global = z_mid + offset
            
            # Check if this plane is in local slice
            if z_global >= z_slice.start and z_global < z_slice.stop:
                z_local = z_global - z_slice.start
                u_z = uhat_2d[..., :, :, z_local]
                energy_z = ((u_z * u_z.conjugate()).real).sum(axis=0)
                
                if energy_sum is None:
                    energy_sum = energy_z
                else:
                    energy_sum += energy_z
                n_contributing_planes += 1
        
        # Reduce across MPI ranks
        if n_contributing_planes > 0:
            energy_2d = energy_sum / n_contributing_planes
        else:
            energy_2d = numpy.zeros(uhat_2d.shape[1:-1])
        
        # All ranks participate in reduction
        total_planes = uhat.grid.comm.reduce(n_contributing_planes, MPI.SUM)
        energy_2d = uhat.grid.comm.reduce(energy_2d, MPI.SUM)
        
        if uhat.grid.comm.rank == 0:
            # Average across all ranks' contributions
            energy_2d /= total_planes if total_planes > 0 else 1
            
            # Compute spectrum
            dk = numpy.sqrt(uhat.grid.dk[0] * uhat.grid.dk[1])
            kappa, spectrum = self.integrate_shell_2d(energy_2d, dk, uhat.grid)
            
            # Calculate L
            if len(kappa) > 1 and numpy.any(spectrum[1:] > 0):
                kappa_pos = kappa[1:]
                spec_pos = spectrum[1:]
                numerator = numpy.trapz(spec_pos / kappa_pos, kappa_pos)
                denominator = numpy.trapz(spec_pos, kappa_pos)
                L = numerator / denominator if denominator > 0 else 0.0
            else:
                L = 0.0
            
            # Write output
            self.outfile.write(f"# t = {time}\n")
            self.outfile.write(f"# L = {L}\n")
            self.outfile.write(f"# Averaged over {total_planes} z-planes around midplane\n")
            for k, s in zip(kappa, spectrum):
                self.outfile.write(f"{k} {s}\n")
            self.outfile.write("\n\n")
            self.outfile.flush()


'''class PlaneSpectra(Diagnostic):
    r"""A diagnostic class for 2D velocity spectra on the midplane (z=0)
    This diagnostic computes the two-dimensional energy spectrum function
    on the z=0 midplane using the existing distributed 2D FFT infrastructure.
    The full domain is transformed and then z=0 is extracted locally from
    each rank's portion.
    
    The spectrum is defined as:
    .. math::
        E(\kappa)
        = \hat{u}_i(\boldsymbol{\kappa}) \hat{u}_i^*(\boldsymbol{\kappa})
    where :math:`i \in \{u, v\}` sums over the in-plane velocity components
    and :math:`\boldsymbol{\kappa} = (k_x, k_y)` is the 2D wavevector.
    The spectrum is azimuthally averaged to produce a 1D spectrum as a
    function of :math:`\kappa = |\boldsymbol{\kappa}|`.
    The integral length scale is computed according to:
    .. math::
        L = \frac{\int_0^\infty \kappa^{-1} E(\kappa) \, d\kappa}{
                \int_0^\infty E(\kappa) \, d\kappa}
    """
    
    def integrate_shell_2d(self, energy, dk, grid):
        """Integrate energy over annular shells in 2D wavevector space
        
        Bins energy by 2D wavenumber magnitude and applies appropriate
        azimuthal averaging. Shell area element is 2*pi*kappa*dkappa.
        """
        nbins = int(grid.kmax2D/dk) + 1
        spectrum = numpy.zeros([nbins])
        count = numpy.zeros([nbins], dtype=int)
        
        # Iterate over 2D wavenumber magnitudes and energy
        for kappa, e in numpy.nditer([grid.kmag_2D, energy], flags=['refs_ok']):
            if kappa > 0:  # Exclude kappa=0
                idx = int(kappa/dk)
                if idx < nbins:
                    spectrum[idx] += e
                    count[idx] += 1
        
        k = numpy.arange(nbins) * dk
        
        # Reduce across MPI ranks
        spectrum = grid.comm.reduce(spectrum)
        count = grid.comm.reduce(count)
        
        if grid.comm.rank == 0:
            # Normalize: multiply by 2*pi*kappa / (count * dk^2)
            # This gives the energy density per unit wavenumber
            denom = count * dk**2
            denom[denom == 0] = 1  # Avoid division by zero
            spectrum = 2 * numpy.pi * k * spectrum / denom
        
        return k, spectrum
    
    def diagnostic(self, time, equations, uhat):
        """Write the 2D midplane spectrum to a file"""
        # Get u and v velocities in physical space
        u = uhat[:2].to_physical()
        
        # Transform full domain using 2D FFT (x-y only, z distributed)
        # Result has shape [2, kx, ky, nz_local] where nz is distributed
        uhat_2d = u.to_spectral_2Dxy()
        
        # Check if this rank has z=0 in its local slice
        z0_global = uhat.grid.pdims[2] // 2  # Midplane index
        z_slice = uhat.grid._local_z_slice
        has_z0 = (z0_global >= z_slice.start and z0_global < z_slice.stop)
        
        if has_z0:
            # Extract local z-index of z=0 plane
            z0_local = z0_global - z_slice.start
            # Get 2D energy at z=0: |u_hat|^2 + |v_hat|^2
            # Shape: [2, kx, ky] -> sum over components -> [kx, ky]
            u_z0 = uhat_2d[..., :, :, z0_local]
            energy_2d = ((u_z0 * u_z0.conjugate()).real).sum(axis=0)
        else:
            # This rank doesn't have z=0, contribute zeros
            k2d_shape = uhat_2d.shape[1:-1]  # [kx, ky]
            energy_2d = numpy.zeros(k2d_shape)
        
        # 2D wavenumber spacing in x-y plane
        dk = numpy.sqrt(uhat.grid.dk[0] * uhat.grid.dk[1])
        
        # Perform azimuthal shell integration
        kappa, spectrum = self.integrate_shell_2d(energy_2d, dk, uhat.grid)
        
        # Calculate integral length scale and write output
        if uhat.grid.comm.rank == 0:
            # Exclude kappa=0 (first bin)
            if len(kappa) > 1 and numpy.any(spectrum[1:] > 0):
                kappa_pos = kappa[1:]
                spec_pos = spectrum[1:]
                
                # Numerical integration using trapezoidal rule
                numerator = numpy.trapz(spec_pos / kappa_pos, kappa_pos)
                denominator = numpy.trapz(spec_pos, kappa_pos)
                
                L = numerator / denominator if denominator > 0 else 0.0
            else:
                L = 0.0
            
            # Write output
            self.outfile.write(f"# t = {time}\n")
            self.outfile.write(f"# L = {L}\n")
            for k, s in zip(kappa, spectrum):
                self.outfile.write(f"{k} {s}\n")
            self.outfile.write("\n\n")
            self.outfile.flush()'''


class FieldDump(Diagnostic):
    """Full spectral field file dumps"""
    def __init__(self, filename="data{:04g}", **kwargs):
        super().__init__(**kwargs)
        self.filename=filename
    
    def diagnostic(self, time, equations, uhat):
        """Write the solution fields in MPI format"""
        uhat.checkpoint(self.filename.format(time))


class VTKDump(Diagnostic):
    """VTK file dumps

    This :class:`Diagnostic` class dumps full fields in physical space
    using VTK format.  A list of the names to use for the fields must
    be passed as *names*.  An optional *filename* pattern can be
    passed, which will be formatted using Python string formatting
    (:meth:`str.format`), with the value of the current timestep set
    to *time*.
    """
    def __init__(self, names, filename="./rank{rank}_phys{time:04g}", **kwargs):
        # We defer evtk import so it does not become a dependency
        # unless we are actually using the VTKDump diagnostic.
        # If including a file path, use the absolute path.
        import evtk
        self.gridToVTK = evtk.hl.gridToVTK
        self.writeParallelVTKGrid = evtk.hl.writeParallelVTKGrid
        super().__init__(**kwargs)
        self.filename = filename
        self.names = names

    def diagnostic(self, time, equations, uhat):
        u = numpy.asarray(uhat.to_physical())
        # Grid awareness
        x_ind = uhat.grid.comm.rank % uhat.grid.decomp[0]
        y_ind = int(uhat.grid.comm.rank / uhat.grid.decomp[0])
        r = uhat.grid.comm.rank

        # Get rank to left and send to left
        sendx2 = (r-1)%uhat.grid.decomp[0] + ((r)//uhat.grid.decomp[0]) * uhat.grid.decomp[0]
        getxfrom = (r+1)%uhat.grid.decomp[0] + ((r)//uhat.grid.decomp[0]) * uhat.grid.decomp[0]
        # Handle if sending to another rank versus sending to self
        if sendx2 != r:
            x_crossover = numpy.empty_like(u[:, 0:1, :, :])
            x_crossover = numpy.ascontiguousarray(x_crossover)
            x_send = u[:, 0:1, :, :]
            x_send = numpy.ascontiguousarray(x_send)
            uhat.grid.comm.Sendrecv(x_send, dest=sendx2, recvbuf=x_crossover, source=getxfrom)
            u = numpy.concatenate((u, x_crossover), axis=1)
        else:
            u = numpy.concatenate((u, u[:, 0:1, :, :]), axis=1)
        
        # Get rank on top and send to top
        sendy2 = (r+uhat.grid.decomp[0])%uhat.grid.comm.size
        getyfrom = (r-uhat.grid.decomp[0])%uhat.grid.comm.size
        # Handle if sending to another rank versus sending to self
        if sendy2 != r:
            y_crossover = numpy.empty_like(u[:, :, 0:1, :])
            y_crossover = numpy.ascontiguousarray(y_crossover)
            y_send = u[:, :, 0:1, :]
            y_send = numpy.ascontiguousarray(y_send)
            uhat.grid.comm.Sendrecv(y_send, dest=sendy2, recvbuf=y_crossover, source=getyfrom)
            u = numpy.concatenate((u, y_crossover), axis=2)
        else:
            u = numpy.concatenate((u, u[:, :, 0:1, :]), axis=2)

        u = numpy.concatenate((u, u[:,:,:,0:1]), axis=3)
        
        coord = uhat.grid.x
        dx = uhat.grid.dx
        x = coord[0, :, 0, 0]
        y = coord[1, 0, :, 0]
        z = coord[2, 0, 0, :]
        x = numpy.concatenate((x, numpy.array([x[-1]+dx[0]])))
        y = numpy.concatenate((y, numpy.array([y[-1]+dx[1]])))
        z = numpy.concatenate((z, numpy.array([z[-1]+dx[2]])))

        # Round to prevent issues with extent caused by numerical precision
        start = (numpy.round(x[0]/dx[0]), numpy.round(y[0]/dx[1]), numpy.round(z[0]/dx[2]))
        end = (numpy.round(x[-1]/dx[1]), numpy.round(y[-1]/dx[1]), numpy.round(z[-1]/dx[2]))

        from pathlib import Path
        path = Path(self.filename)
        directory = path.parent
        file = path.name
        sub_dir = Path(f"{directory}/vtk_t{time}")
        sub_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{sub_dir}/{file}"

        self.gridToVTK(
            filename.format(time=time, rank=uhat.grid.comm.rank),
            x, y, z, 
            pointData = dict(zip(self.names, u)),
            start=start,
        )
        recvbuf_start = None
        recvbuf_end = None

        if uhat.grid.comm.rank == 0:
            recvbuf_start = numpy.empty((uhat.grid.comm.size, 3), dtype=float)
            recvbuf_end = numpy.empty((uhat.grid.comm.size, 3), dtype=float)
        uhat.grid.comm.Gather(numpy.array(start), recvbuf_start, root=0)
        uhat.grid.comm.Gather(numpy.array(end), recvbuf_end, root=0)
        if uhat.grid.comm.rank == 0:
            starts = [tuple(row) for row in recvbuf_start]
            ends = [tuple(row) for row in recvbuf_end]
            self.writeParallelVTKGrid(self.filename.format(time=time, rank=uhat.grid.comm.rank),
                                coordsData=(tuple(uhat.grid.pdims+1), x.dtype),
                                starts = starts,
                                ends=ends,
                                sources=[filename.format(time=time, rank=rank_) + ".vtr" for rank_ in range(uhat.grid.comm.size)],
                                pointData={
                                    i: (u.dtype, 1) for i in self.names
                                },
            )
