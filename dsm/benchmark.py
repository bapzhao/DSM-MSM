"""Original four-method timing calculation; plotting code is omitted."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
import csv
import gc
import math
import time
from pathlib import Path
import numpy as np
from numpy.fft import fft, ifft


def wave_number(n: int, length: float = 2.0 * np.pi) -> np.ndarray:
    half = n // 2
    k = np.zeros(n, dtype=float)
    dk = 2.0 * np.pi / length
    k[:half] = np.arange(half, dtype=float) * dk
    k[half] = -half * dk
    k[half + 1:] = np.arange(-half + 1, 0, dtype=float) * dk
    return k


def kint(i: int, n: int) -> int:
    half = n // 2
    if i == 0:
        return 0
    if (n % 2 == 0) and (i == half):
        return -half
    return i if i < half else i - n


def random_hermitian_uk(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uk = np.zeros(n, dtype=complex)
    half = n // 2
    for m in range(1, half):
        amp = (1.0 + float(m)) ** (-5.0 / 6.0)
        amp *= 0.8 + 0.4 * rng.random()
        phase = rng.uniform(-np.pi, np.pi)
        val = amp * np.exp(1j * phase)
        uk[m] = val
        uk[n - m] = np.conj(val)
    uk[0] = 0.0
    uk[half] = 0.0
    norm = math.sqrt(float(np.sum(np.abs(uk) ** 2)))
    if norm > 0.0:
        uk /= norm
    return uk


def smag_eps_target(uk: np.ndarray, k: np.ndarray, cs: float, delta: float) -> float:
    n = uk.size
    dudx = ifft(1j * k * uk).real * n
    nu_t = (cs * delta) ** 2 * np.abs(dudx)
    eps_sgs = float(np.mean(nu_t * dudx**2))
    return -eps_sgs


def unpadded_rhs(uk: np.ndarray, k: np.ndarray) -> np.ndarray:
    n = uk.size
    u = ifft(uk).real * n
    uk2 = fft(u * u) / n
    return -0.5j * k * uk2


def advance_unpadded(uk: np.ndarray, k: np.ndarray, nu: float, dt: float) -> np.ndarray:
    out = uk + dt * unpadded_rhs(uk, k)
    out = out - nu * (k**2) * out * dt
    out[out.size // 2] = 0.0
    return out


def pad_spectrum(uk: np.ndarray, padded_n: int) -> np.ndarray:
    n = uk.size
    half = n // 2
    up = np.zeros(padded_n, dtype=complex)
    up[:half] = uk[:half]
    up[-(half - 1):] = uk[half + 1:]
    return up


def truncate_spectrum(up: np.ndarray, n: int) -> np.ndarray:
    half = n // 2
    uk = np.zeros(n, dtype=complex)
    uk[:half] = up[:half]
    uk[half + 1:] = up[-(half - 1):]
    return uk


def dealiased_3_2_rhs(uk: np.ndarray, k: np.ndarray) -> np.ndarray:
    n = uk.size
    padded_n = 3 * n // 2
    up = pad_spectrum(uk, padded_n)
    u = ifft(up).real * padded_n
    up2 = fft(u * u) / padded_n
    uk2 = truncate_spectrum(up2, n)
    return -0.5j * k * uk2


def advance_3_2(uk: np.ndarray, k: np.ndarray, nu: float, dt: float) -> np.ndarray:
    out = uk + dt * dealiased_3_2_rhs(uk, k)
    out = out - nu * (k**2) * out * dt
    out[out.size // 2] = 0.0
    return out


def alias_power_fft(uk: np.ndarray, k: np.ndarray) -> float:
    rhs = unpadded_rhs(uk, k)
    half = uk.size // 2
    return float(2.0 * np.real(np.vdot(uk[1:half], rhs[1:half])))


def shifted_uk(uk: np.ndarray, k: np.ndarray, alpha: float) -> np.ndarray:
    return uk * np.exp(1j * k * alpha)


def pq_two_shift_fft(uk: np.ndarray, k: np.ndarray) -> tuple[float, float]:
    n = uk.size
    alpha0 = 0.0
    alpha1 = np.pi / (2.0 * n)
    y0 = alias_power_fft(shifted_uk(uk, k, alpha0), k)
    y1 = alias_power_fft(shifted_uk(uk, k, alpha1), k)
    return y0, y1


def active_indices(k: np.ndarray, n: int) -> np.ndarray:
    kc = n // 2 - 1
    return np.where((np.abs(k) >= 1.0) & (np.abs(k) <= float(kc)))[0]


def direct_triad_pq_loop(uk: np.ndarray, k: np.ndarray) -> tuple[float, float]:
    n = uk.size
    half = n // 2
    active = active_indices(k, n)
    active_mask = np.zeros(n, dtype=bool)
    active_mask[active] = True
    amp = np.abs(uk)
    phi = np.angle(uk)
    P = 0.0
    Q = 0.0

    for k_idx in range(1, half):
        kk = kint(k_idx, n)
        ak = float(amp[k_idx])
        if ak == 0.0:
            continue
        for m_idx in active:
            n_idx = (k_idx - m_idx) % n
            if not active_mask[n_idx]:
                continue
            if amp[m_idx] == 0.0 or amp[n_idx] == 0.0:
                continue

            km = kint(int(m_idx), n)
            kn = kint(int(n_idx), n)
            delta = km + kn - kk
            if delta == n:
                sigma = 1.0
            elif delta == -n:
                sigma = -1.0
            else:
                continue

            B = abs(k[k_idx]) * amp[k_idx] * amp[m_idx] * amp[n_idx]
            Phi = phi[m_idx] + phi[n_idx] - phi[k_idx]
            P += B * np.sin(Phi)
            Q += sigma * B * np.cos(Phi)
    return float(P), float(Q)


def solve_alpha_zero_or_clip(P: float, Q: float, eps_target: float, n: int) -> float:
    amp = math.hypot(P, Q)
    if not np.isfinite(amp) or amp < 1.0e-30:
        return 0.0
    phi = math.atan2(P, Q)
    y = float(np.clip(eps_target / amp, -1.0, 1.0))
    theta = math.asin(y)
    candidates = [theta, math.pi - theta]
    period = 2.0 * math.pi / n
    alphas = []
    for th in candidates:
        a = (th - phi) / n
        a = a - round(a / period) * period
        alphas.append(a)
    return min(alphas, key=abs)


def step_double_alpha_fft(uk: np.ndarray, k: np.ndarray, nu: float, dt: float, cs: float) -> np.ndarray:
    eps_target = 0
    # eps_target = smag_eps_target(uk, k, cs, 2.0 * np.pi / uk.size)

    P, Q = pq_two_shift_fft(uk, k)
    alpha = solve_alpha_zero_or_clip(P, Q, eps_target, uk.size)
    return advance_unpadded(shifted_uk(uk, k, alpha), k, nu, dt)


def step_direct_triad_pq(uk: np.ndarray, k: np.ndarray, nu: float, dt: float, cs: float) -> np.ndarray:
    eps_target = 0
    # eps_target = smag_eps_target(uk, k, cs, 2.0 * np.pi / uk.size)

    P, Q = direct_triad_pq_loop(uk, k)
    alpha = solve_alpha_zero_or_clip(P, Q, eps_target, uk.size)
    return advance_unpadded(shifted_uk(uk, k, alpha), k, nu, dt)


def step_3_2(uk: np.ndarray, k: np.ndarray, nu: float, dt: float, cs: float) -> np.ndarray:
    del cs
    return advance_3_2(uk, k, nu, dt)


def step_unpadded(uk: np.ndarray, k: np.ndarray, nu: float, dt: float, cs: float) -> np.ndarray:
    del cs
    return advance_unpadded(uk, k, nu, dt)


def median_time(func, uk: np.ndarray, k: np.ndarray, nu: float, dt: float, cs: float,
                min_time: float, max_repeats: int) -> tuple[float, int]:
    for _ in range(2):
        func(uk.copy(), k, nu, dt, cs)

    samples = []
    repeats_done = 0
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        while sum(samples) < min_time and repeats_done < max_repeats:
            u = uk.copy()
            t0 = time.perf_counter()
            func(u, k, nu, dt, cs)
            samples.append(time.perf_counter() - t0)
            repeats_done += 1
    finally:
        if gc_was_enabled:
            gc.enable()
    return float(np.median(samples)), repeats_done


def benchmark(sizes: list[int], min_time: float, max_repeats: int) -> list[dict[str, float]]:
    methods = [
        ("unpadded_fft", "no dealiasing", step_unpadded),
        ("dealias_3_2", "3/2 padding-truncation method", step_3_2),
        ("double_alpha_fft", "DSM", step_double_alpha_fft),
        ("direct_triad_pq", "Direct folded-triad method", step_direct_triad_pq),
    ]
    rows = []
    for n in sizes:
        k = wave_number(n)
        uk = random_hermitian_uk(n, seed=20260505 + n)
        nu = 5.0e-3
        dt = 2.0e-4
        cs = 0.2
        for method_id, label, func in methods:
            elapsed, repeats = median_time(func, uk, k, nu, dt, cs, min_time, max_repeats)
            rows.append({
                "M": n,
                "method_id": method_id,
                "method": label,
                "time_s": elapsed,
                "repeats": repeats,
            })
            print(f"M={n:5d} {method_id:18s} {elapsed:.6e} s  repeats={repeats}")
    return rows


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["M", "method_id", "method", "time_s", "repeats"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32,48,64,96,128,192,256,384,512])
    parser.add_argument("--min-time", type=float, default=0.15)
    parser.add_argument("--max-repeats", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(n < 8 or n % 2 for n in args.sizes) or args.max_repeats < 1 or args.min_time <= 0:
        parser.error("Sizes must be even and >= 8; repeats and minimum time must be positive.")
    if args.output.exists():
        parser.error("Output exists; choose a new file.")
    rows = benchmark(args.sizes, args.min_time, args.max_repeats)
    write_csv(rows, args.output)
