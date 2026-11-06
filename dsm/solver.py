"""Fourier pseudospectral kernels used by the one-dimensional Burgers driver.

Import this module through dsm/run_burgers.py; there is no standalone run or
file output. NumPy is the only external dependency. The driver supplies the
initial condition, phase shifts, retained-mode mask, and output handling.

Arrays contain normalized coefficients uk = fft(u)/M in NumPy FFT ordering;
the real velocity is ifft(uk).real*M. spe_k contains physical wavenumbers.
A True entry in dealiasing_mask removes that coefficient. The driver chooses
the sampling grid and mask for aliased or 3/2-padded calculations.

advance_step applies explicit convection followed by molecular viscosity.
advance_step_les_smag_non_average inserts an explicit Smagorinsky update between
those operations. Each update uses the full dt. The published freely decaying
cases use forcek=None and do not activate the optional forcing branches.
"""
import numpy as np
from numpy.fft import fft, ifft
from numpy import cos, sin
from random import uniform
PI = np.pi


def WaveNumber(X,nb_X):
    """Return physical wavenumbers for domain length X and nb_X grid points.

    The returned array follows NumPy FFT ordering, with negative wavenumbers
    in its second half. nb_X counts grid points, not retained positive modes.
    """
    
    #spectral space grid
    spe_i=np.zeros(nb_X)
    spe_k=np.zeros(nb_X)
    dk=2*PI/X
    for i in range(0,int((nb_X-1)/2)+1):
        spe_i[i]=i
        spe_k[i]=spe_i[i]*dk

    for i in range(int((nb_X-1)/2)+1,nb_X):
        spe_i[i]=i-nb_X
        spe_k[i]=spe_i[i]*dk
    
    return spe_k


def advance_step(uk,spe_k,NU,dt,dealiasing_mask=None,forcek=None):
    """Advance convection and molecular viscosity by one explicit split step.

    uk is the normalized Fourier velocity, spe_k its wavenumbers, NU the
    kinematic viscosity, and dt the time step. Supply a Boolean mask whose
    True entries discard unresolved coefficients of the nonlinear product.
    The returned array is the updated Fourier velocity.
    """
    u_temp=np.zeros_like(uk)
    u2_temp=np.zeros_like(uk)
    energy = np.sum((np.abs(uk)**2))

    # compute convection term
    u_temp=ifft(uk).real*np.size(uk)
    u2_temp=u_temp*u_temp
    uk2_temp= fft(u2_temp)/np.size(uk)
    if dealiasing_mask is not None:
        uk2_temp = dealiasing(uk2_temp,dealiasing_mask)
    else:
        uk2_temp = dealiasing(uk2_temp)
    uk = uk - 0.5*1j*spe_k*uk2_temp*dt
    uk = uk - NU*(spe_k**2)*uk*dt
    # First-order explicit convection and viscosity substeps.
    if(forcek is not None):
        dissp = energy - np.sum((np.abs(uk)**2))
        ksi = uniform(-PI/2, PI/2)
        factor = dissp / (2 * (np.abs(uk[forcek])**2))
        factor = - cos(ksi) + np.sqrt(cos(ksi)**2 + factor)
        xxx = abs(uk[forcek])**2
        uk[forcek] = uk[forcek] * (1 + factor * (cos(ksi) + 1j * sin(ksi)))
        uk[-forcek] = uk[-forcek] * (1 + factor * (cos(ksi) - 1j * sin(ksi)))
    return uk


def advance_step_les_smag_non_average(uk, spe_k, NU, dt,
                         dealiasing_mask=None,
                         forcek=None,
                         Cs=0.17,
                         Delta=None,
                         cap_nu_sgs_average=None,
                         enforce_filter=True):
    """
    Advance the 1D Burgers LES equation with explicit split substeps.

    LES equation (filtered):
        d ubar/dt + d/dx(ubar^2/2) = NU d2 ubar/dx2 + d/dx( nu_sgs * d ubar/dx )
        nu_sgs = (Cs*Delta)^2 * |d ubar/dx|

    Notes:
    - Fourier normalization:
        u = ifft(uk).real * N ;  uk = fft(u)/N
    - Nonlinear products are evaluated in physical space; the supplied spectral
      mask is applied to their Fourier coefficients.
    """

    N = np.size(uk)

    # ---- (optional) enforce LES filter: keep only resolved modes in uk itself
    if enforce_filter and (dealiasing_mask is not None):
        uk = uk.copy()
        uk[dealiasing_mask] = 0.0

    # =========================
    # 1) Convection: uk <- uk - dt * d/dx(u^2/2)
    # =========================
    u = ifft(uk).real * N
    u2 = u * u
    uk2 = fft(u2) / N

    if dealiasing_mask is not None:
        uk2 = dealiasing(uk2, dealiasing_mask)
    else:
        uk2 = dealiasing(uk2)

    uk = uk - 0.5 * 1j * spe_k * uk2 * dt

    # ---- (optional) filter again after convection
    if enforce_filter and (dealiasing_mask is not None):
        uk[dealiasing_mask] = 0.0

    # =========================
    # 2) Smagorinsky SGS: uk <- uk + dt * d/dx( nu_sgs * du/dx )
    # =========================
    if (Delta is not None) and (Cs is not None) and (Cs != 0.0):
        # du/dx in physical space
        dudx = ifft(1j * spe_k * uk).real * N

        # eddy viscosity
        nu_sgs = (Cs * Delta)**2 * np.abs(dudx)

        tau = nu_sgs * dudx
        tau_k = fft(tau) / N

        # dealias the SGS tau term (nonlinear)
        if dealiasing_mask is not None:
            tau_k = dealiasing(tau_k, dealiasing_mask)
        else:
            tau_k = dealiasing(tau_k)

        # add + d/dx(tau)
        uk = uk + 1j * spe_k * tau_k * dt

        if enforce_filter and (dealiasing_mask is not None):
            uk[dealiasing_mask] = 0.0

    # =========================
    # 3) Molecular viscosity: uk <- uk - NU k^2 uk dt
    # =========================
    uk = uk - NU * (spe_k**2) * uk * dt

    if enforce_filter and (dealiasing_mask is not None):
        uk[dealiasing_mask] = 0.0

    # =========================
    # 4) Optional forcing; disabled in the freely decaying cases.
    # =========================
    if(forcek is not None):
        energy_before = np.sum((np.abs(uk)**2))
        dissp = energy_before - np.sum((np.abs(uk)**2))
        ksi = uniform(-PI/2, PI/2)
        factor = dissp / (2 * (np.abs(uk[forcek])**2))
        factor = - cos(ksi) + np.sqrt(cos(ksi)**2 + factor)
        uk[forcek]  = uk[forcek]  * (1 + factor * (cos(ksi) + 1j * sin(ksi)))
        uk[-forcek] = uk[-forcek] * (1 + factor * (cos(ksi) - 1j * sin(ksi)))

    return uk


def dealiasing(uk,dealiasing_mask=None):
    """Zero masked coefficients in place and return uk; a mask is required."""
    if dealiasing_mask is not None:
        uk[dealiasing_mask]=0
    else:
        raise RuntimeError("A dealiasing_mask must be provided.")
    return uk
