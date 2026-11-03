"""Burgers update kernels extracted from the original calculation code."""
import numpy as np
from numpy.fft import fft, ifft
from numpy import cos, sin
from random import uniform
PI = np.pi


def WaveNumber(X,nb_X):
    '''
    input:
        X: domain length
        nb_X: number of domain
    output:
        spe_k: wave number array
    '''
    
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
    u_temp=np.zeros_like(uk)
    u2_temp=np.zeros_like(uk)
    energy = np.sum((np.abs(uk)**2))

    # compute convection term
    u_temp=ifft(uk).real*np.size(uk)
    # print(ifft(uk).imag)
    u2_temp=u_temp*u_temp
    uk2_temp= fft(u2_temp)/np.size(uk)
    if dealiasing_mask is not None:
        uk2_temp = dealiasing(uk2_temp,dealiasing_mask)
    else:
        uk2_temp = dealiasing(uk2_temp)
    uk = uk - 0.5*1j*spe_k*uk2_temp*dt
    uk = uk - NU*(spe_k**2)*uk*dt
    # dudx = ifft(1j * spe_k * uk).real * np.size(uk)
    # u = ifft(uk).real * np.size(uk)
    # ududx = u * dudx
    # uk2_temp = fft(ududx) / np.size(uk)
    # if dealiasing_mask is not None:
    #     uk2_temp = dealiasing(uk2_temp,dealiasing_mask)
    # else:
    #     uk2_temp = dealiasing(uk2_temp)
    # uk = uk - uk2_temp*dt
    #time advance : Euler first order

    # print(f"energy lost after convection: {energy-energybis:2e}, Viscous lost: {energybis-energybisbis:2e}")
    if(forcek is not None):
        dissp = energy - np.sum((np.abs(uk)**2))
        # seed(tm.time_ns() % 1000000)
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
    1D Burgers LES (Smagorinsky) based on your original Euler-splitting step.

    LES equation (filtered):
        d ubar/dt + d/dx(ubar^2/2) = NU d2 ubar/dx2 + d/dx( nu_sgs * d ubar/dx )
        nu_sgs = (Cs*Delta)^2 * |d ubar/dx|

    Notes:
    - Normalization consistent with your code:
        u = ifft(uk).real * N ;  uk = fft(u)/N
    - All nonlinear products are done in physical space and dealiased in spectral space.
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

        # # optional cap to avoid too stiff explicit diffusion
        # if cap_nu_sgs_average is not None:
        #     nu_sgs_average = np.minimum(nu_sgs_average, cap_nu_sgs_average)

        # tau = nu_sgs_average * dudx
        tau = nu_sgs * dudx
        # eps_sgs = nu_sgs_average * np.mean(dudx**2)
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
    # 4) Your forcing (unchanged)
    # =========================
    if(forcek is not None):
        energy_before = np.sum((np.abs(uk)**2))
        # 这里你原逻辑是用 dissip 来调幅；我不改
        dissp = energy_before - np.sum((np.abs(uk)**2))
        ksi = uniform(-PI/2, PI/2)
        factor = dissp / (2 * (np.abs(uk[forcek])**2))
        factor = - cos(ksi) + np.sqrt(cos(ksi)**2 + factor)
        uk[forcek]  = uk[forcek]  * (1 + factor * (cos(ksi) + 1j * sin(ksi)))
        uk[-forcek] = uk[-forcek] * (1 + factor * (cos(ksi) - 1j * sin(ksi)))

    return uk


def dealiasing(uk,dealiasing_mask=None):
    if dealiasing_mask is not None:
        uk[dealiasing_mask]=0
    else:
        raise RuntimeError("请提供 dealiasing_mask 进行去混叠。")
        # nb_X = np.size(uk)/2
        # uk[int(1*nb_X/3):int(2*nb_X/3)] = 0
    return uk
