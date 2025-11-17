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
        """print(numpy.sum(numpy.abs(u[0])))
        print(numpy.sum(numpy.abs(u[1])))
        print(numpy.sum(numpy.abs(u[2])))
        print(u.shape)"""
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
            #print("\nRank " + str(uhat.grid.comm.rank) + "  k:\n" + str(k))
            #print("\nRank " + str(uhat.grid.comm.rank) + "  spectrum:\n" + str(spectrum))
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
        print(spectrum)
        exit()
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
        u_plane = u[:, :, :, int(u.shape[3]/2)]
        uhat2d = numpy.fft.fft2(u_plane, axes=(-2, -1), norm="forward")
        N = uhat.grid.sdims
        i0 = numpy.array([*range(0, N[0]//2+1), *range(-((N[0]-1)//2), 0)])
        uhat2d = uhat2d[3, i0[:, None], i0]
        k, spectrum = self.integrate_shell(
            0.5 *  (uhat2d*uhat2d.conjugate()).real, 
            numpy.prod(uhat.grid.dk[:2])**(1/2), 
            uhat.grid)
        print("How you doin")
        print(spectrum)
        exit()
        for i, s in zip(k, spectrum):
            self.outfile.write("{} {}\n".format(i, s))
        self.outfile.write("\n\n")
        self.outfile.flush()


class Spectra2D_Parallel(Diagnostic):
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
        ispectrum = grid.comm.reduce(ispectrum)
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
        
        if uhat.grid.comm.rank == 0:
            for i, s in zip(k, spectrum):
                self.outfile.write("{} {}\n".format(i, s))
            self.outfile.write("\n\n")
            self.outfile.flush()


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

    .. note::

       :class:`VTKDump` uses the :mod:`evtk` module, which does not
       work for parallel runs (multiple MPI ranks).
    """
    def __init__(self, names, filename="./phys{time:04g}", **kwargs):
        if kwargs['grid'].comm.size != 1:
            warnings.warn(
                "VTKDump does not work with multiple MPI ranks.",
                RuntimeWarning
                )
        # We defer evtk import so it does not become a dependency
        # unless we are actually using the VTKDump diagnostic.
        import evtk
        self.gridToVTK = evtk.hl.gridToVTK
        super().__init__(**kwargs)
        self.filename = filename
        self.names = names

    def diagnostic(self, time, equations, uhat):
        u = numpy.asarray(uhat.to_physical())
        self.gridToVTK(
            self.filename.format(time=time),
            uhat.grid.x[0],
            uhat.grid.x[1],
            uhat.grid.x[2],
            pointData = dict(zip(self.names, u))
            )


class VTKDump_Parallel(Diagnostic):
    """VTK file dumps in parallel

    This :class:`Diagnostic` class dumps full fields in physical space
    using VTK format.  A list of the names to use for the fields must
    be passed as *names*.  An optional *filename* pattern can be
    passed, which will be formatted using Python string formatting
    (:meth:`str.format`), with the value of the current timestep set
    to *time*.

    .. note::

       :class:`VTKDump` uses the :mod:`evtk` module, which does not
       work for parallel runs (multiple MPI ranks).
    """
    def __init__(self, names, filename="./rank{rank}_phys{time:04g}", **kwargs):
        if kwargs['grid'].comm.size != 1:
            warnings.warn(
                "VTKDump does not work with multiple MPI ranks.",
                RuntimeWarning
                )
        # We defer evtk import so it does not become a dependency
        # unless we are actually using the VTKDump diagnostic.
        
        super().__init__(**kwargs)
        self.filename = filename
        self.names = names

    def diagnostic(self, time, equations, uhat):
        import pyvista as pv
        u = numpy.asarray(uhat.to_physical())
        print("Gaining Grid Awareness")
        # Grid awareness
        x_ind = uhat.grid.comm.rank % uhat.grid.decomp[0]
        y_ind = int(uhat.grid.comm.rank / uhat.grid.decomp[0])
        r = uhat.grid.comm.rank
        # Get rank to left and send to left
        sendx2 = (r-1)%uhat.grid.decomp[0] + ((r)//uhat.grid.decomp[0]) * uhat.grid.decomp[0]
        getxfrom = (r+1)%uhat.grid.decomp[0] + ((r)//uhat.grid.decomp[0]) * uhat.grid.decomp[0]
        if sendx2 != r:
            x_crossover = numpy.empty_like(u[:, 0:1, :, :])
            x_crossover = numpy.ascontiguousarray(x_crossover)
            x_send = u[:, 0:1, :, :]
            x_send = numpy.ascontiguousarray(x_send)
            uhat.grid.comm.Sendrecv(x_send, dest=sendx2, recvbuf=x_crossover, source=getxfrom)
            u = numpy.concatenate((u, x_crossover), axis=1)
            print("Complete")
        else:
            u = numpy.concatenate((u, u[:, 0:1, :, :]), axis=1)
            print("All by myself, don't communicate")
        sendy2 = (r+uhat.grid.decomp[0])%uhat.grid.comm.size
        
        exit()
        """if sendx2 != r:
            print(f"Rank {uhat.grid.comm.rank} sending x to {sendx2} with tag {r}")
            uhat.grid.comm.send(u[:, 0:1, :, :], dest=sendx2, tag=r)
            print("After send")
            exit()"""
        
        # Receive from rank to right
        getxfrom = (r+1)%uhat.grid.decomp[0] + ((r)//uhat.grid.decomp[0]) * uhat.grid.decomp[0]
        if getxfrom != r:
            print(f"Rank {uhat.grid.comm.rank} receiving x from {getxfrom} with tag {(r+1)%uhat.grid.decomp[0] + ((r)//uhat.grid.decomp[0]) * uhat.grid.decomp[0]}")
            x_crossover = uhat.grid.comm.recv(source=getxfrom, tag=(r+1)%uhat.grid.decomp[0] + ((r)//uhat.grid.decomp[0]) * uhat.grid.decomp[0])
            u = numpy.concatenate((u, x_crossover), axis=1)
        else:
            #print("same processor x")
            u = numpy.concatenate((u, u[:, 0:1, :, :]), axis=1)

        # Get rank to the top and send to top
        sendy2 = (r+uhat.grid.decomp[0])%uhat.grid.comm.size
        if sendy2 != r:
            #print(f"Rank {uhat.grid.comm.rank} sending y to {sendy2} with tag {uhat.grid.comm.size+r}")
            uhat.grid.comm.send(u[:, :, 0:1, :], dest=sendy2, tag=uhat.grid.comm.size+r)

        # Receive from rank on bottom
        getyfrom = (r-uhat.grid.decomp[0])%uhat.grid.comm.size
        if getyfrom != r:
            #print(f"Rank {uhat.grid.comm.rank} receiving y from {(r-uhat.grid.decomp[0])%uhat.grid.comm.size} with tag {(r-uhat.grid.decomp[0])%uhat.grid.comm.size+uhat.grid.comm.size}")
            y_crossover = uhat.grid.comm.recv(source=getyfrom, tag=(r-uhat.grid.decomp[0])%uhat.grid.comm.size+uhat.grid.comm.size)
            u = numpy.concatenate((u, y_crossover), axis=2)
        else:
            #print("same processor y")
            u = numpy.concatenate((u, u[:, :, 0:1, :]), axis=2)
            

        u = numpy.concatenate((u, u[:,:,:,0:1]), axis=3)
        #print("I am grid aware")
        
        coord = uhat.grid.x
        x = coord[0, :, 0, 0]
        y = coord[1, 0, :, 0]
        z = coord[2, 0, 0, :]
        
        # create raw coordinate grid
        grid_ijk = numpy.mgrid[
            x[0] : x[-1] + 2*(uhat.grid.box_size[0]/uhat.grid.pdims[0]) : uhat.grid.box_size[0]/(uhat.grid.pdims[0]),#x.shape[0]+1,
            y[0] : y[-1] + 2*(uhat.grid.box_size[1]/uhat.grid.pdims[1]) : uhat.grid.box_size[1]/(uhat.grid.pdims[1]),#y.shape[0]+1,
            z[0] : z[-1] + 2*(uhat.grid.box_size[2]/uhat.grid.pdims[2]) : uhat.grid.box_size[2]/(uhat.grid.pdims[2]) #z.shape[0]+1
        ]

        # repeat array along each Cartesian axis for connectivity
        for axis in range(1, 4):
            grid_ijk = grid_ijk.repeat(2, axis=axis)
            u = u.repeat(2, axis=axis)

        # slice off unnecessarily doubled edge coordinates
        grid_ijk = grid_ijk[:, 1:-1, 1:-1, 1:-1]
        u = u[:, 1:-1, 1:-1, 1:-1]

        # reorder and reshape to VTK order
        corners = grid_ijk.transpose().reshape(-1, 3)
        corner_values = u.transpose().reshape(-1,4)

        dims = numpy.array([len(x), len(y), len(z)]) + 1
        grid = pv.ExplicitStructuredGrid(dims, corners)
        
        '''for i in range(len(self.names)):
            if i<len(corner_values[0,:]):
                grid.point_data[self.names[i]] = corner_values[:, i]
            else:
                break'''
        grid = grid.compute_connectivity()
        for i in range(len(self.names)):
            if i<len(corner_values[0,:]):
                grid.point_data[self.names[i]] = corner_values[:, i]
            else:
                print("UHOH")
                break
        #grid.plot(show_edges=True)
        filename=self.filename.format(rank=uhat.grid.comm.rank, time=time)
        grid.save(filename+".vtu")
        """if uhat.grid.comm.rank == 0:
        #    pv.MultiBlock([self.filename.format(rank=r, time=time)+".vtk" for r in range(uhat.grid.comm.size)]).save('out.pvtu')
            from xml.etree.ElementTree import Element, SubElement, tostring
            from xml.dom import minidom

            def write_pvtu(piece_files, output="out.pvtu", datatype="Float32"):
                vtkfile = Element("VTKFile", type="PUnstructuredGrid", version="0.1", byte_order="LittleEndian")
                punstructured = SubElement(vtkfile, "PUnstructuredGrid")
                # Describe your point and cell data layout here if needed
                ppoints = SubElement(punstructured, "PPoints")
                SubElement(ppoints, "PDataArray",
                    type=datatype,
                    NumberOfComponents="3",
                    Name="Points")
                for f in piece_files:
                    SubElement(punstructured, "Piece", Source=f)

                xml_str = minidom.parseString(tostring(vtkfile)).toprettyxml(indent="  ")
                with open(output, "w") as fh:
                    fh.write(xml_str)
            
            files = [self.filename.format(rank=r, time=time)+".vtu" for r in range(uhat.grid.comm.size)]
            write_pvtu(files)
        exit()"""


        

'''class GrowthRate(Diagnostic):
    """Calculated the growth rate"""
    def diagnostic(self, time, equations, uhat):
        """ubar, u = uhat.to_physical().disturbance()
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
            self.outfile.flush()"""

        c = self.uhat.to_physical()[3]
        c_light = -1
        c_heavy = 1
        mean_vals = numpy.mean(c)
        X = '''

'''class Lske(Diagnostic):
    def __init__(self, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def integrate_shell(self, u, dk, grid):
        N = grid.sdims
        nbins = int(numpy.sqrt((N[0]//2)**2 + (N[1]//2)**2)) + 1
        energy = numpy.zeros([nbins, u.shape[-1]])
        ienergy = numpy.zeros([nbins], dtype=int)
        kmag = grid.kmag_2D#/(numpy.sqrt(numpy.prod(grid.dk[:2])))
        for i in range(len(kmag[:,0])):
            for j in range(len(kmag[0,:])):
                k = kmag[i,j]
                energy[int(k/dk), :] += u[i,j,:] * dk
                ienergy[int(k/dk)] += 1
        #for k, v in numpy.nditer([grid.kmag_2D, u]):
        #    energy[int(k/dk), :] +=v
        #    ienergy[int(k/dk)] += 1
        k = numpy.arange(nbins)/self.scale#*dk
        #print(k)
        #print(energy.shape)
        #print(ienergy)
        #print(numpy.sum(energy[1:len(energy[:,0])], axis=0))
        energy /= (ienergy[:,None])#/2)/numpy.prod(grid.dk[:2])
        #print(numpy.sum(energy[1:len(energy[:,0])]/k[1:len(k),None], axis=0))
        #print(energy)
        #print(numpy.nan_to_num(energy/k[:,None], nan=0)[:,0])
        #return numpy.nan_to_num(numpy.sum(numpy.nan_to_num(energy[1:len(energy[:,0])]/k[1:len(k),None]*dk, nan=0), axis=0)/numpy.sum(energy*dk, axis=0), nan=0)
        return 3*numpy.pi/4*numpy.sum(energy[1:len(energy[:,0])]/k[1:len(k),None], axis=0) / numpy.sum(energy[1:len(energy[:,0])], axis=0)

    def diagnostic(self, time, equations, uhat):
        u = uhat.to_physical()
        #uhat2d = numpy.fft.fft2(u, axes=(1,2), norm="forward")
        u1 = u[0]
        v1 = u[1]
        uhat2d = numpy.fft.fft2(u1, axes=(0,1), norm="forward")
        vhat2d = numpy.fft.fft2(v1, axes=(0,1), norm="forward")
        N = uhat.grid.sdims
        i0 = numpy.array([*range(0, N[0]//2+1), *range(-((N[0]-1)//2), 0)])
        uhat2d_spec = uhat2d[i0[:,None], i0, :]
        vhat2d_spec = vhat2d[i0[:,None], i0, :]
        E = 0.5 * (uhat2d_spec*uhat2d_spec.conjugate() + vhat2d_spec*vhat2d_spec.conjugate()).real
        #uhat2d_spec = uhat2d[:, i0[:,None], i0, :]
        #u_hat = uhat2d_spec[0,:,:,:]
        #v_hat = uhat2d_spec[1,:,:,:]
        #E = 0.5 * (u_hat*u_hat.conjugate() + v_hat*v_hat.conjugate()).real
        #print(numpy.sum(numpy.sum(E, axis=0), axis=1))
        l_int = self.integrate_shell(E, numpy.prod(uhat.grid.dk[:2])**(1/2), uhat.grid)
        z = uhat.grid.x[2,0,0,:]
        for i, s in zip(z, l_int):
            self.outfile.write("{} {}\n".format(i, s))
        self.outfile.write("\n\n")
        self.outfile.flush()
        
    
class Lvd(Diagnostic):
    def __init__(self, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def integrate_shell(self, u, dk, grid):
        N = grid.sdims
        nbins = int(numpy.sqrt((N[0]//2)**2 + (N[1]//2)**2)) + 1
        energy = numpy.zeros([nbins, u.shape[-1]])
        ienergy = numpy.zeros([nbins], dtype=int)
        kmag = grid.kmag_2D
        for i in range(len(kmag[:,0])):
            for j in range(len(kmag[0,:])):
                k = kmag[i,j]
                energy[int(k/dk), :] += u[i,j,:] * dk
                ienergy[int(k/dk)] += 1
        k = numpy.arange(nbins)/self.scale
        energy /= (ienergy[:,None]/2)/numpy.prod(grid.dk[:2])
        return 3*numpy.pi/4*numpy.sum(energy[1:len(energy[:,0])]/k[1:len(k),None], axis=0) / numpy.sum(energy[1:len(energy[:,0])], axis=0)

    def diagnostic(self, time, equations, uhat):
        u = uhat.to_physical()
        u1 = u[0]
        v1 = u[1]
        c1 = u[3]
        uchat2d = numpy.fft.fft2(u1*numpy.abs(c1)**0.5 * numpy.sign(c1), axes=(0,1), norm="forward")
        vchat2d = numpy.fft.fft2(v1*numpy.abs(c1)**0.5 * numpy.sign(c1), axes=(0,1), norm="forward")
        N = uhat.grid.sdims
        i0 = numpy.array([*range(0, N[0]//2+1), *range(-((N[0]-1)//2), 0)])
        uchat2d_spec = uchat2d[i0[:,None], i0, :]
        vchat2d_spec = vchat2d[i0[:,None], i0, :]
        E = 0.5 * (uchat2d_spec*uchat2d_spec.conjugate() + vchat2d_spec*vchat2d_spec.conjugate()).real
        l_int = self.integrate_shell(E, numpy.prod(uhat.grid.dk[:2])**(1/2), uhat.grid)
        z = uhat.grid.x[2,0,0,:]
        for i, s in zip(z, l_int):
            self.outfile.write("{} {}\n".format(i, s))
        self.outfile.write("\n\n")
        self.outfile.flush()

class L_int_mid(Diagnostic):
    def __init__(self, scale=1, dr=0.1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.dr = 0.1

    def integrate_shell(self, u, dk, grid):
        return 0

    def diagnostic(self, time, equations, uhat):
        u_phys = uhat.to_physical()
        x = uhat.grid.x[0,:,0,0]
        y = uhat.grid.x[1,0,:,0]
        z = uhat.grid.x[2,0,0,:]
        u = u_phys[0,:,:,z.shape[0]//2]
        v = u_phys[1,:,:,z.shape[0]//2]
        vel = numpy.sqrt(u**2 + v**2)
        vrms2 = numpy.mean(u**2 + v**2)
        counter_dict = {}
        value_dict = {}
        shiftx = numpy.arange(-u.shape[0]//2, u.shape[0]//2)
        shifty = numpy.arange(-v.shape[0]//2, v.shape[0]//2)
        for sx in shiftx:
            x_rolled = numpy.roll(x, sx, axis=0)
            for sy in shifty:
                u_rolled = numpy.roll(numpy.roll(u, sx, axis=0), sy, axis=1)
                v_rolled = numpy.roll(numpy.roll(v, sx, axis=0), sy, axis=1)
                y_rolled = numpy.roll(y, sy, axis=0)

                r = round(numpy.sqrt((x[0]-x_rolled[0])**2 + (y[0]-y_rolled[0])**2) / self.dr) * self.dr
                theta = numpy.arctan2(y_rolled, x_rolled)
                vel_rolled = numpy.sqrt((u_rolled * numpy.cos(theta))**2 + (v_rolled * numpy.sin(theta))**2)
                counter_dict[r]  = counter_dict.get(r,0 ) + 1
                value_dict[r]  = value_dict.get(r,0 ) + vel_rolled * vel
        f = numpy.mean([value_dict[k] / counter_dict[k] for k in counter_dict] / vrms2, axis=(1,2))
        f = numpy.nan_to_num(f, nan=0.0)
        
        l_int = numpy.sum(f)*self.dr
        print("Integral Length Scale " + str(l_int))
        
        #for i, s in zip(z, l_int):
        #    self.outfile.write("{} {}\n".format(i, s))
        self.outfile.write("{}\n".format(l_int))
        self.outfile.write("\n\n")
        self.outfile.flush()

class L_int_AE525(Diagnostic):
    """def __init__(self, scale=1, dr=0.1, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale"""

    def diagnostic(self, time, equations, uhat):
        u_phys = uhat.to_physical()
        x = uhat.grid.x[0,:,0,0]
        y = uhat.grid.x[1,0,:,0]
        z = uhat.grid.x[2,0,0,:]
        u = u_phys[0,:,:,z.shape[0]//2]
        v = u_phys[1,:,:,z.shape[0]//2]

        N = x.shape[0]

        Rxx = numpy.sum(u*u)/N**2
        Ryy = numpy.sum(v*v)/N**2
        Rxy = numpy.sum(u*v)/N**2
        vrms = numpy.sqrt(1/2 * (Rxx+Ryy))

        f = numpy.zeros(N)
        
        for i in range(0, N):
            for j in range(0, N):
                for l in range(0, N):
                    i_l = i+l
                    j_l = j+l
                    if i_l >= N:
                        i_l = i_l-N
                    if j_l >=N:
                        j_l = j_l-N
                    f[l] += u[i,j]*u[i_l,j] + v[i,j]*v[i,j_l]
        
        r = numpy.linspace(0, x[N-1], N)
        f /= (N**2 * 2 * vrms**2)
        l_int = numpy.trapz(f, r)

        self.outfile.write("{} {}\n".format(time, l_int))
        self.outfile.write("\n\n")
        self.outfile.flush()

class L_int(Diagnostic):
    def diagnostic(self, time, equations, uhat):
        u_phys = uhat.to_physical()
        x = uhat.grid.x[0,:,0,0]
        y = uhat.grid.x[1,0,:,0]
        z = uhat.grid.x[2,0,0,:]
        u = u_phys[0,:,:,z.shape[0]//2]
        v = u_phys[1,:,:,z.shape[0]//2]

        dx = x[1] - x[0]
        N = x.size

        def autocorr_1d(field):
            f = numpy.zeros(N)
            for l in range(N):
                f[l] = numpy.mean(field[:, :-l] * field[:, l:]) if l > 0 else numpy.mean(field**2)
                """if l ==1 and time >0:
                    print(field[:,:-l])
                    print(field[:,l:])
                    exit()"""
            return f / f[0]

        fu = autocorr_1d(u)
        fv = autocorr_1d(v)
        f = 0.5 * (fu + fv)

        l_int = numpy.trapz(f, dx=dx)

        self.outfile.write("{} {}\n".format(time, l_int))
        self.outfile.write("\n\n")
        self.outfile.flush()'''


class L_int(Diagnostic):
    def diagnostic(self, time, equations, uhat):
        u_phys = uhat.to_physical()
        x = uhat.grid.x[0,:,0,0]   # length Nx
        y = uhat.grid.x[1,0,:,0]   # length Ny
        z = uhat.grid.x[2,0,0,:]
        u = u_phys[0,:,:, z.shape[0]//2 ]   # shape (Nx, Ny)
        v = u_phys[1,:,:, z.shape[0]//2 ]

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        Nx, Ny = u.shape

        def autocorr_along_axis(field, axis=0):
            N = field.shape[axis]
            f = numpy.zeros(N)
            for l in range(N):
                f[l] = numpy.mean(field * numpy.roll(field, -l, axis=axis))
            return f

        # Autocorrelations of components along x and y
        fu_x = autocorr_along_axis(u, axis=0)   # u correlated along x
        fv_x = autocorr_along_axis(v, axis=0)   # v correlated along x
        fu_y = autocorr_along_axis(u, axis=1)   # u correlated along y
        fv_y = autocorr_along_axis(v, axis=1)   # v correlated along y

        # Normalize each by its zero-lag value
        fu_x /= fu_x[0]
        fv_x /= fv_x[0]
        fu_y /= fu_y[0]
        fv_y /= fv_y[0]
        zero_cross_fu_x = numpy.where(fu_x <=0)[0]
        zero_cross_fv_y = numpy.where(fv_y <=0)[0]
        if zero_cross_fu_x.size ==0:
            r0x = len(fu_x)
        else:
            r0x = zero_cross_fu_x[0]
        if zero_cross_fv_y.size ==0:
            r0y = len(fv_y)
        else:
            r0y = zero_cross_fv_y[0]
        # Average components if you want a single curve per direction
        #f_x = 0.5*(fu_x + fv_x)   # average along x-direction
        #f_y = 0.5*(fu_y + fv_y)   # average along y-direction
        #f = 0.5*(fu_x + fv_x)

        # Integrate to get integral length scales
        r_x = numpy.arange(Nx)*dx
        r_y = numpy.arange(Ny)*dy
        L_x = numpy.trapz(fu_x[:r0x], r_x[:r0x])
        L_y = numpy.trapz(fv_y[:r0y], r_y[:r0y])
        #l_int = numpy.trapz(f, r_x)
        l_int = (L_x + L_y)
        self.outfile.write("{} {}\n".format(time, l_int))
        self.outfile.write("\n\n")
        self.outfile.flush()

        #print("L_x (correlation along x):", L_x)
        #print("L_y (correlation along y):", L_y)

        # A simple isotropic estimate (average of two directions)
        #L_iso_est = 0.5*(L_x + L_y)
        #print("Isotropic estimate (avg):", L_iso_est)