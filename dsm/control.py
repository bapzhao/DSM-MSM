# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yicong Zhao
"""Power diagnostics and phase selection for the one-dimensional DSM driver.

This module is imported by dsm/run_burgers.py; it is not a simulation entry.
Inputs are one-dimensional normalized Fourier coefficients, u_hat = fft(u)/M,
and matching wavenumbers on the periodic domain [0, 2*pi). The freely decaying
cases use real velocity fields and omit the mean and Nyquist modes.

phase_model returns P, Q, A for power(alpha) = P*cos(M*alpha) + Q*sin(M*alpha).
solve_alpha returns the selected shift, clipped power target, and saturation flag.
The shift is in physical-coordinate units, so translation multiplies mode k by
exp(1j*k*alpha).

Sign convention: alias_power and the target accepted by solve_alpha are signed
kinetic-energy rates. Negative power removes energy; the positive dissipation
used in the manuscript and driver history is its negative. Smagorinsky targets
returned here also use the signed-power convention. No files are written.
"""
from __future__ import annotations
import numpy as np
DOMAIN_LENGTH = 2.0 * np.pi


def code_energy(state: np.ndarray) -> float:
    """Return the sum of squared Fourier amplitudes used in solver output."""
    return float(np.sum(np.abs(state) ** 2))


def translate(state: np.ndarray, wavenumber: np.ndarray, alpha: float) -> np.ndarray:
    """Return coefficients of u(x + alpha), without changing the input array."""
    return state * np.exp(1j * wavenumber * alpha)


def kinetic_energy(state: np.ndarray) -> float:
    """Return the spatially averaged kinetic energy, 0.5*sum(abs(u_hat)**2)."""
    return 0.5 * code_energy(state)


def viscous_dissipation(state: np.ndarray, wavenumber: np.ndarray, viscosity: float) -> float:
    """Return positive mean molecular dissipation, viscosity*mean(u_x**2)."""
    gradient = np.fft.ifft(1j * wavenumber * state).real * state.size
    return float(viscosity * np.mean(gradient**2))


def smagorinsky_power_target(
    state: np.ndarray,
    wavenumber: np.ndarray,
    coefficient: float,
) -> float:
    """Return negative Smagorinsky dissipation with filter width 2*pi/M."""
    gradient = np.fft.ifft(1j * wavenumber * state).real * state.size
    delta = DOMAIN_LENGTH / state.size
    dissipation = np.mean((coefficient * delta) ** 2 * np.abs(gradient) * gradient**2)
    return -float(dissipation)


def hermitian_error(state: np.ndarray) -> float:
    """Measure relative conjugate-symmetry error, excluding mean and Nyquist."""
    half = state.size // 2
    numerator = np.max(np.abs(state[1:half] - np.conj(state[-1:half:-1])))
    denominator = max(float(np.max(np.abs(state))), 1.0e-300)
    return float(numerator / denominator)


def alias_power(state: np.ndarray, wavenumber: np.ndarray) -> float:
    """Return mean convective power, including both modes of each conjugate pair.

    The ordinary resolved triads have zero net power. The remaining net rate
    is therefore the aliasing contribution; positive values add kinetic energy.
    """
    physical = np.fft.ifft(state).real * state.size
    nonlinear_hat = np.fft.fft(physical**2) / state.size
    rhs = -0.5j * wavenumber * nonlinear_hat
    half = state.size // 2
    return float(2.0 * np.real(np.vdot(state[1:half], rhs[1:half])))


def phase_model(state: np.ndarray, wavenumber: np.ndarray) -> tuple[float, float, float]:
    """Return P, Q, A from shifts 0 and pi/(2*M) of the same input field."""
    size = state.size
    p_value = alias_power(state, wavenumber)
    q_value = alias_power(translate(state, wavenumber, np.pi / (2.0 * size)), wavenumber)
    return p_value, q_value, float(np.hypot(p_value, q_value))


def solve_alpha(
    p_value: float,
    q_value: float,
    target: float,
    size: int,
    previous: float | None,
    amplitude_tolerance: float = 1.0e-30,
) -> tuple[float, float, bool]:
    """Choose a shift for the requested signed power, clipped to [-A, A].

    Returns (alpha, clipped_target, saturated). If previous is supplied, select
    the sine branch and periodic copy closest to that previous shift. For
    negligible amplitude, retain the previous shift or use zero.
    """
    amplitude = float(np.hypot(p_value, q_value))
    if not np.isfinite(amplitude) or amplitude < amplitude_tolerance:
        alpha = 0.0 if previous is None else previous
        return alpha, 0.0, abs(target) > amplitude

    clipped_target = float(np.clip(target, -amplitude, amplitude))
    phase = float(np.arctan2(p_value, q_value))
    theta_1 = float(np.arcsin(clipped_target / amplitude))
    theta_2 = float(np.pi - theta_1)
    period = 2.0 * np.pi / size
    candidates = []
    for theta in (theta_1, theta_2):
        base = (theta - phase) / size
        if previous is None:
            candidate = base - np.round(base / period) * period
        else:
            candidate = base + np.round((previous - base) / period) * period
        candidates.append(float(candidate))
    if previous is None:
        errors = [abs(p_value * np.cos(size * a) + q_value * np.sin(size * a) - clipped_target) for a in candidates]
    else:
        errors = [abs(a - previous) for a in candidates]
    return candidates[int(np.argmin(errors))], clipped_target, abs(target) > amplitude
