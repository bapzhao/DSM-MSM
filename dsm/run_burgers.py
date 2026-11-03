#!/usr/bin/env python3
"""Portable calculation entry for the manuscript's one-dimensional cases."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

import control
import solver


CASES = ("dns-reference", "zero-reference", "zero-dsm", "zero-raw",
         "dsm-smag", "dsm-filtered", "smag", "coarse")


def initial_state(cutoff, size, seed=2):
    """Original k_L=2 spectrum, Python random phases, K(0)=1/2."""
    k = solver.WaveNumber(2*np.pi, size)
    spectrum = k**4 * np.exp(-0.5*k*k)
    spectrum /= spectrum.sum()
    state = np.zeros(size, complex)
    rng = random.Random(seed)
    for m in range(1, cutoff):
        phase = rng.uniform(0, 2*np.pi)
        state[m] = np.sqrt(spectrum[m]) * (np.cos(phase)-1j*np.sin(phase))
        state[-m] = state[m].conjugate()
    return np.fft.fft(np.fft.ifft(state).real*size)/size


def project(state, cutoff, size):
    if cutoff > len(state)//2:
        raise ValueError("Cannot project to more modes than the source contains.")
    out = np.zeros(size, complex)
    out[1:cutoff] = state[1:cutoff]
    out[-1:-cutoff:-1] = state[-1:-cutoff:-1]
    return out


def convection(state, k, cutoff):
    physical = np.fft.ifft(state).real*len(state)
    rhs = -0.5j*k*np.fft.fft(physical**2)/len(state)
    rhs[np.abs(k) >= cutoff] = 0
    return rhs


def load_restart(path, cutoff, size, expected_time):
    with np.load(path, allow_pickle=False) as data:
        state = data["u_hat"]
        time = float(data["time"])
    if not np.isclose(time, expected_time, rtol=0, atol=1e-12):
        raise ValueError(f"Restart time {time} differs from --start {expected_time}.")
    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError("Invalid restart Fourier coefficients.")
    return project(state, cutoff, size)


def load_target(path, start, stop):
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if not {"t", "epsilon_sgs"}.issubset(data.dtype.names or ()):
        raise ValueError("Target CSV must contain t,epsilon_sgs columns.")
    t, values = data["t"], data["epsilon_sgs"]
    if len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(values).all():
        raise ValueError("Invalid target history.")
    if np.any(np.diff(t) <= 0) or t[0] > start+1e-12 or t[-1] < stop-1e-12:
        raise ValueError("Target times must increase and cover the full requested interval.")
    return t, values


def save_state(directory, state, time, cutoff):
    np.savez_compressed(directory/f"state_t{time:.6f}.npz",
                        u_hat=state, time=time, cutoff=cutoff)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--cutoff", type=int)
    p.add_argument("--nu", type=float)
    p.add_argument("--dt", type=float, default=2e-4)
    p.add_argument("--start", type=float)
    p.add_argument("--stop", type=float)
    p.add_argument("--cs", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--restart", type=Path)
    p.add_argument("--target", type=Path)
    p.add_argument("--filter-cutoff", type=int, default=32)
    p.add_argument("--snapshot-times", nargs="*", type=float,
                   default=[0.0, 0.2, 0.3, 0.338, 0.4, 0.6, 0.8])
    p.add_argument("--output", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    zero = a.case.startswith("zero-")
    is_reference = a.case == "dns-reference"
    a.cutoff = a.cutoff if a.cutoff is not None else (512 if zero or is_reference else 32)
    a.nu = a.nu if a.nu is not None else (0.001 if zero else 0.005)
    a.start = a.start if a.start is not None else (0.0 if zero or is_reference else 0.3)
    a.stop = a.stop if a.stop is not None else (0.8002 if is_reference else 0.8)
    if a.cutoff < 4 or a.nu < 0 or a.cs < 0 or a.dt <= 0 or a.stop <= a.start:
        p.error("Require cutoff>=4, nu>=0, cs>=0, dt>0, stop>start.")
    steps = (a.stop-a.start)/a.dt
    if abs(steps-round(steps)) > 1e-8:
        p.error("stop-start must be an integer multiple of dt.")
    if not 2 <= a.filter_cutoff <= a.cutoff and is_reference:
        p.error("filter-cutoff must be between 2 and cutoff.")
    if a.start > 0 and a.restart is None:
        p.error("A nonzero start requires --restart; first run dns-reference.")
    if a.case == "dsm-filtered" and a.target is None:
        p.error("dsm-filtered requires --target from dns-reference.")
    if a.output is None and not a.dry_run:
        p.error("--output is required.")
    if a.output is not None and a.output.exists():
        p.error("Output directory exists; choose a new directory.")
    if a.nu*(a.cutoff-1)**2*a.dt >= 2:
        p.error("Explicit molecular-viscosity stability factor nu*kmax^2*dt must be <2.")
    return a


def run(a):
    padded = a.case in ("dns-reference", "zero-reference", "smag", "coarse")
    size = (3 if padded else 2)*a.cutoff
    k = solver.WaveNumber(2*np.pi, size)
    mask = np.abs(k) >= a.cutoff
    state = (initial_state(a.cutoff, size, a.seed) if a.restart is None else
             load_restart(a.restart, a.cutoff, size, a.start))
    target = load_target(a.target, a.start, a.stop) if a.case == "dsm-filtered" else None
    a.output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(a).items()}
    (a.output/"configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    steps = round((a.stop-a.start)/a.dt)
    selected_steps = {round((t-a.start)/a.dt) for t in a.snapshot_times
                      if a.start-1e-12 <= t <= a.stop+1e-12
                      and abs((t-a.start)/a.dt-round((t-a.start)/a.dt)) < 1e-8}
    selected_steps |= {0, steps}
    previous = None
    filtered_history = []
    with (a.output/"history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "K", "Dnu", "epsilon_target", "epsilon_applied",
                         "alias_amplitude", "saturated", "alpha"])
        for n in range(steps+1):
            time = a.start+n*a.dt
            energy = control.kinetic_energy(state)
            dnu = control.viscous_dissipation(state, k, a.nu)
            if not np.isfinite(state).all() or not np.isfinite(energy):
                raise FloatingPointError(f"Nonfinite solution at t={time}; existing output is partial.")
            if n in selected_steps:
                save_state(a.output, state, time, a.cutoff)
            if a.case == "dns-reference":
                active = np.abs(k) < a.filter_cutoff
                kf = float(0.5*np.sum(np.abs(state[active])**2))
                df = float(a.nu*np.sum(np.abs(k[active]*state[active])**2))
                filtered_history.append((time, kf, df))
            desired = amplitude = applied = alpha = 0.0
            saturated = False
            shifted = state
            if a.case in ("zero-dsm", "dsm-smag", "dsm-filtered"):
                if a.case == "dsm-smag":
                    desired = -control.smagorinsky_power_target(state, k, a.cs)
                elif a.case == "dsm-filtered":
                    desired = float(np.interp(time, *target))
                p, q, amplitude = control.phase_model(state, k)
                alpha, _, saturated = control.solve_alpha(p, q, -desired, size, previous)
                previous = alpha
                shifted = control.translate(state, k, alpha)
                applied = -control.alias_power(shifted, k)
                expected = np.clip(desired, -amplitude, amplitude)
                if abs(applied-expected) > 2e-10*max(1.0, abs(desired), amplitude):
                    raise RuntimeError("DSM phase reconstruction failed.")
            elif a.case == "smag":
                gradient = np.fft.ifft(1j*k*state).real*size
                stress = (a.cs*np.pi/a.cutoff)**2*np.abs(gradient)*gradient
                stress_hat = np.fft.fft(stress)/size
                stress_hat[mask] = 0
                applied = -float(np.vdot(state, 1j*k*stress_hat).real)
                desired = applied
            writer.writerow([time, energy, dnu, desired, applied, amplitude,
                             int(saturated), alpha])
            if n == steps:
                break
            if a.case == "smag":
                state = solver.advance_step_les_smag_non_average(
                    state, k, a.nu, a.dt, mask, Cs=a.cs, Delta=np.pi/a.cutoff)
            else:
                state = solver.advance_step(shifted, k, a.nu, a.dt, mask)
    if filtered_history:
        arr = np.asarray(filtered_history)
        np.savetxt(a.output/"filtered_history.csv", arr, delimiter=",",
                   header="t,K_filtered,Dnu_filtered", comments="")
        epsilon = -np.diff(arr[:,1])/a.dt-arr[:-1,2]
        np.savetxt(a.output/"sgs_target.csv", np.column_stack((arr[:-1,0],epsilon)),
                   delimiter=",", header="t,epsilon_sgs", comments="")
    print(f"Completed {a.case}: {steps} steps; output: {a.output}")


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value
                          for key,value in vars(args).items()}, indent=2))
    else:
        run(args)
