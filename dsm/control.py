"""Evaluate net aliasing power and select the DSM translation phase."""
from __future__ import annotations
import numpy as np
DOMAIN_LENGTH = 2.0 * np.pi


def code_energy(state: np.ndarray) -> float:
    """Return the sum of squared Fourier amplitudes used in solver output."""
    return float(np.sum(np.abs(state) ** 2))


def translate(state: np.ndarray, wavenumber: np.ndarray, alpha: float) -> np.ndarray:
    return state * np.exp(1j * wavenumber * alpha)


def kinetic_energy(state: np.ndarray) -> float:
    return 0.5 * code_energy(state)


def viscous_dissipation(state: np.ndarray, wavenumber: np.ndarray, viscosity: float) -> float:
    gradient = np.fft.ifft(1j * wavenumber * state).real * state.size
    return float(viscosity * np.mean(gradient**2))


def smagorinsky_power_target(
    state: np.ndarray,
    wavenumber: np.ndarray,
    coefficient: float,
) -> float:
    gradient = np.fft.ifft(1j * wavenumber * state).real * state.size
    delta = DOMAIN_LENGTH / state.size
    dissipation = np.mean((coefficient * delta) ** 2 * np.abs(gradient) * gradient**2)
    return -float(dissipation)


def hermitian_error(state: np.ndarray) -> float:
    half = state.size // 2
    numerator = np.max(np.abs(state[1:half] - np.conj(state[-1:half:-1])))
    denominator = max(float(np.max(np.abs(state))), 1.0e-300)
    return float(numerator / denominator)


def alias_power(state: np.ndarray, wavenumber: np.ndarray) -> float:
    """Net convective power over positive modes; true triads sum to zero."""
    physical = np.fft.ifft(state).real * state.size
    nonlinear_hat = np.fft.fft(physical**2) / state.size
    rhs = -0.5j * wavenumber * nonlinear_hat
    half = state.size // 2
    return float(2.0 * np.real(np.vdot(state[1:half], rhs[1:half])))


def phase_model(state: np.ndarray, wavenumber: np.ndarray) -> tuple[float, float, float]:
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
