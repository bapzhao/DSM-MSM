"""Run the two-dimensional incompressible Navier--Stokes comparison cases.

The domain is the periodic square [0, 2*pi)^2. The msm method controls net
aliasing power at zero using the diagonal shift alpha_x = alpha_y. The reference
method uses 3/2 padding; raw uses the unshifted, aliased nonlinear product.
All three use the same initial-condition prescription and Dedalus RK222 scheme.

Usage (from the repository root, in the nsdedalus environment)
-----------------------------------------------------------
    python msm/run_ns.py --method msm --output outputs/msm
    python msm/run_ns.py --method reference --output outputs/msm_reference
    python msm/run_ns.py --method raw --output outputs/msm_raw
    mpiexec -n 4 python msm/run_ns.py --method msm --output outputs/msm_mpi

Use a new output directory. Set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1
before launching. --kmax sets half the grid size in each direction; --dt is
the maximum CFL-controlled time step. --help lists all options, --dry-run
prints the configuration, and --check-only tests the initial MSM reconstruction
without advancing or writing files. Time advancement requires Dedalus 3.0.5.

Output and rate convention
--------------------------
Each run writes configuration.json and HDF5 snapshots containing velocity,
pressure, vorticity, divergence, kinetic-energy density, and selected shifts.
MPI output may be distributed over process files. Snapshots are saved after
restoring the original coordinate frame. The raw calculation may become unstable.
The eps_* diagnostics use signed domain-integrated convective power. To compare
with positive mean aliasing dissipation in the manuscript, use
epsilon_alias = -power/(Lx*Ly). The prescribed target in this script is zero.
"""


import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--method", choices=("msm", "reference", "raw"), default="msm")
parser.add_argument("--kmax", type=int, default=128)
parser.add_argument("--nu", type=float, default=7.23e-4)
parser.add_argument("--stop", type=float, default=10.0)
parser.add_argument("--dt", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=20260527)
parser.add_argument("--snapshot-every", type=int, default=1000)
parser.add_argument("--output", type=Path)
parser.add_argument("--check-only", action="store_true",
                    help="Initialize the fields and check the MSM reconstruction; write no files.")
parser.add_argument("--dry-run", action="store_true", help="Print configuration without importing Dedalus.")
args = parser.parse_args()
if args.kmax < 4 or args.nu < 0 or args.stop <= 0 or args.dt <= 0 or args.snapshot_every < 1:
    parser.error("Require kmax>=4, nu>=0, stop>0, dt>0, snapshot-every>=1.")
if not args.check_only and not args.dry_run and args.output is None:
    parser.error("--output is required for time advancement.")
if args.output is not None and args.output.exists():
    parser.error("Output directory exists; choose a new output directory.")
if args.dry_run:
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, indent=2))
    raise SystemExit(0)

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

outputPATH = str(args.output) if args.output is not None else None
# Parameters
Lx, Ly = 2*np.pi, 2*np.pi

initial_condition="random"

Kmax=args.kmax
Nx, Ny = 2*Kmax, 2*Kmax

nu=args.nu
dealias = 1.5 if args.method == "reference" else 1.0
stop_sim_time = args.stop
timestepper = d3.RK222
max_timestep = args.dt
k0 = 1
use_formula32_alias_control = args.method == "msm"
eps_target = 0.0
formula32_search_n = 257
dtype = np.complex128

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis, ybasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis))
tau_p = dist.Field(name='tau_p')
psi = dist.Field(name='psi', bases=(xbasis, ybasis))
alpha_x_output = dist.Field(name='alpha_x')
alpha_y_output = dist.Field(name='alpha_y')

x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)

def local_physical_wavenumbers(field):
    """Return local physical kx, ky arrays aligned with field['c']."""
    local_elements = dist.coeff_layout.local_elements(field.domain, scales=1)
    kx_1d = xbasis.wavenumbers[local_elements[0]]
    ky_1d = ybasis.wavenumbers[local_elements[1]]
    return kx_1d[:, None], ky_1d[None, :]

def apply_spectral_translation_to_velocity(u, alphax, alphay):
    """Translate both velocity components in place using physical-coordinate shifts."""
    u.require_coeff_space()
    kx_phase, ky_phase = local_physical_wavenumbers(u)
    phase = np.exp(1j * (kx_phase * alphax + ky_phase * alphay))
    u['c'][0] *= phase
    u['c'][1] *= phase

def velocity_grid_arrays(u):
    """Return local grid-space velocity component copies."""
    u.require_grid_space()
    return np.array(u['g'][0]), np.array(u['g'][1])

def kinetic_energy_mean(ux, uy):
    """Return mean kinetic energy over the supplied velocity-component arrays."""
    return np.mean(0.5 * (np.abs(ux)**2 + np.abs(uy)**2))

def isotropic_energy_spectrum(ux, uy):
    """Bin kinetic energy into nearest-integer radial Fourier shells."""
    ux = np.real(ux)
    uy = np.real(uy)
    nx, ny = ux.shape

    ux_hat = np.fft.fft2(ux) / (nx * ny)
    uy_hat = np.fft.fft2(uy) / (nx * ny)
    energy_2d = 0.5 * (np.abs(ux_hat)**2 + np.abs(uy_hat)**2)

    kx_fft = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx/nx)
    ky_fft = 2.0 * np.pi * np.fft.fftfreq(ny, d=Ly/ny)
    KX, KY = np.meshgrid(kx_fft, ky_fft, indexing="ij")
    shell = np.rint(np.sqrt(KX**2 + KY**2)).astype(int)

    spectrum = np.zeros(shell.max() + 1)
    for kk in range(spectrum.size):
        spectrum[kk] = np.sum(energy_2d[shell == kk])
    return spectrum

def total_convective_energy_rate(u):
    """Return integral of u dot [-(u dot grad)u], the convective KE rate."""
    return d3.integ(-(u @ (u @ d3.grad(u))))

def evaluate_scalar(expr):
    """Evaluate a Dedalus scalar expression and return a real Python float."""
    data = expr.evaluate().allgather_data('g')
    return float(np.real(np.squeeze(data)))

def kinetic_energy_integral(u):
    """Build the Dedalus expression for domain-integrated kinetic energy."""
    return d3.integ(0.5*(u@u))

def copy_velocity(dst, src):
    """Copy local Fourier velocity coefficients into an existing probe field."""
    src.require_coeff_space()
    dst.require_coeff_space()
    dst['c'][:] = src['c']

def epsilon_at_alpha(u_src, u_tmp, alphax, alphay):
    """Evaluate shifted integrated convective power without changing u_src."""
    copy_velocity(u_tmp, u_src)
    apply_spectral_translation_to_velocity(u_tmp, alphax, alphay)
    return evaluate_scalar(total_convective_energy_rate(u_tmp))

def epsilon_at_alphax(u_src, u_tmp, alphax, alphay=0.0):
    """Evaluate the same signed power with an optional fixed y-direction shift."""
    return epsilon_at_alpha(u_src, u_tmp, alphax, alphay)

def formula32_basis(alphax, alphay, Mx, My):
    """
    Basis order:
    X1, X2, X3, X4, Px, Qx, Py, Qy.
    """
    tx = Mx * alphax
    ty = My * alphay
    cx = np.cos(tx)
    sx = np.sin(tx)
    cy = np.cos(ty)
    sy = np.sin(ty)
    return np.array([
        cx*cy,
        sx*sy,
        sx*cy,
        cx*sy,
        cx,
        sx,
        cy,
        sy,
    ], dtype=np.float64)

def formula32_epsilon(coeffs, alphax, alphay, Mx, My):
    """Evaluate the eight-term model using signed integrated-power coefficients."""
    X1, X2, X3, X4, Px, Qx, Py, Qy = coeffs
    tx = Mx * alphax
    ty = My * alphay
    cx = np.cos(tx)
    sx = np.sin(tx)
    cy = np.cos(ty)
    sy = np.sin(ty)
    return (
        X1*cx*cy
        + X2*sx*sy
        + X3*sx*cy
        + X4*cx*sy
        + Px*cx
        + Qx*sx
        + Py*cy
        + Qy*sy
    )

def compute_formula32_coefficients(u_src, u_tmp, Mx, My):
    """
    Reconstruct the eight-coefficient power model from eight full-rank samples.
    """
    phase_pairs = (
        (0.0, 0.0),
        (+np.pi/2.0, 0.0),
        (-np.pi/2.0, 0.0),
        (0.0, +np.pi/2.0),
        (0.0, -np.pi/2.0),
        (+np.pi/2.0, +np.pi/2.0),
        (+np.pi/2.0, -np.pi/2.0),
        (-np.pi/2.0, +np.pi/2.0),
    )
    matrix = []
    values = []
    for tx, ty in phase_pairs:
        alphax = tx / Mx
        alphay = ty / My
        matrix.append(formula32_basis(alphax, alphay, Mx, My))
        values.append(epsilon_at_alpha(u_src, u_tmp, alphax, alphay))

    matrix = np.asarray(matrix, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    rank = np.linalg.matrix_rank(matrix)
    cond = np.linalg.cond(matrix)

    if rank == matrix.shape[1]:
        coeffs = np.linalg.solve(matrix, values)
    else:
        coeffs = np.linalg.lstsq(matrix, values, rcond=None)[0]

    fit = matrix @ coeffs
    info = {
        "q_x": np.pi / (2.0 * Mx),
        "q_y": np.pi / (2.0 * My),
        "rank": rank,
        "cond": cond,
        "fit_linf": np.max(np.abs(fit - values)),
        "eps0": values[0],
    }
    return coeffs, info

def solve_alphaxy_from_formula32(
    coeffs,
    Mx,
    My,
    eps_target,
    alpha_x_prev=None,
    alpha_y_prev=None,
    search_n=257,
):
    """
    Impose the isotropy-consistent diagonal closure alpha_x = alpha_y = alpha.
    The reconstructed model then becomes a one-parameter function
    epsilon_diag(alpha), and alpha is selected by scanning one diagonal phase
    period for epsilon_diag(alpha) = eps_target.
    """
    if search_n < 3:
        raise ValueError("formula32_search_n must be at least 3.")
    if abs(Mx - My) > 1.0e-12 * max(1.0, abs(Mx), abs(My)):
        raise ValueError(
            "The alpha_x=alpha_y diagonal closure assumes an isotropic "
            "sampling grid with Mx == My."
        )

    if alpha_x_prev is None and alpha_y_prev is None:
        center = 0.0
    elif alpha_x_prev is None:
        center = alpha_y_prev
    elif alpha_y_prev is None:
        center = alpha_x_prev
    else:
        center = 0.5 * (alpha_x_prev + alpha_y_prev)

    period_alpha = 2.0 * np.pi / Mx
    alpha_offsets = np.linspace(-0.5 * period_alpha, 0.5 * period_alpha, int(search_n))
    alpha_values = center + alpha_offsets
    eps_values = np.asarray(
        [formula32_epsilon(coeffs, alpha, alpha, Mx, My) for alpha in alpha_values],
        dtype=np.float64,
    )
    residuals = eps_values - eps_target
    eps_min = float(np.min(eps_values))
    eps_max = float(np.max(eps_values))
    scale = max(1.0, abs(eps_target), np.max(np.abs(eps_values)))
    tol = 1.0e-12 * scale
    candidates = []

    def eps_diag(alpha):
        return formula32_epsilon(coeffs, alpha, alpha, Mx, My)

    for index in np.flatnonzero(np.abs(residuals) <= tol):
        alpha = float(alpha_values[index])
        candidates.append((alpha, alpha, float(eps_values[index])))

    for index in range(len(alpha_values) - 1):
        f_left = residuals[index]
        f_right = residuals[index + 1]
        if abs(f_left) <= tol or abs(f_right) <= tol:
            continue
        if f_left * f_right > 0.0:
            continue

        left = float(alpha_values[index])
        right = float(alpha_values[index + 1])
        for _ in range(60):
            mid = 0.5 * (left + right)
            f_mid = eps_diag(mid) - eps_target
            if abs(f_mid) <= tol:
                left = right = mid
                break
            if (eps_diag(left) - eps_target) * f_mid <= 0.0:
                right = mid
            else:
                left = mid
        alpha = 0.5 * (left + right)
        candidates.append((alpha, alpha, float(eps_diag(alpha))))

    if candidates:
        distances = [
            (alpha - center)**2
            for alpha, _, _ in candidates
        ]
        best = int(np.argmin(distances))
        alphax, alphay, predicted = candidates[best]
    else:
        index = int(np.argmin(np.abs(residuals)))
        alpha = float(alpha_values[index])
        alphax = alpha
        alphay = alpha
        predicted = float(eps_values[index])

    info = {
        "eps_min": eps_min,
        "eps_max": eps_max,
        "eps_range": eps_max - eps_min,
        "target_inside": int(bool(candidates)),
        "best_error": abs(predicted - eps_target),
    }
    return alphax, alphay, predicted, info



# Problem
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - u@grad(u)")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

if initial_condition == "random":
    # Clear psi
    psi['g'] = 0
    psi.require_coeff_space()
    psi['c'][:] = 0.0

    # Dedalus local physical Fourier wavenumbers in coefficient space.
    kx, ky = local_physical_wavenumbers(psi)
    # Random streamfunction initial condition
    kmin = 1
    kmax = Kmax*dealias
    kp = 12
    psi_amp = 1e-2
    seed = args.seed
    alphax =np.pi / 4
    alphay = np.pi / 6


    k2 = kx**2 + ky**2
    kmag = np.sqrt(k2)

    # Active spectral band
    active = (kmag >= kmin) & (kmag <= kmax) & (k2 > 0)
    positive_half = (kx > 0) | ((kx == 0) & (ky > 0))
    kx_canon = np.where(positive_half, kx, -kx)
    ky_canon = np.where(positive_half, ky, -ky)
    phase_seed = (
        12.9898 * kx_canon
        + 78.233 * ky_canon
        + 0.137 * seed
    )
    theta0 = 2.0 * np.pi * (
        np.sin(phase_seed) * 43758.5453123
        - np.floor(np.sin(phase_seed) * 43758.5453123)
    )
    theta = np.where(positive_half, theta0, -theta0)
    amp_psi = np.zeros_like(kmag, dtype=np.float64)
    amp_psi[active] = (
        psi_amp
        * kmag[active]
        * np.exp(-(kmag[active] / kp)**2)
    )
    psi['c'][active] = amp_psi[active] * np.exp(1j * theta[active])
    psi['c'][~active] = 0.0
    psi['c'][k2 == 0] = 0.0

    # Transform streamfunction to grid space
    psi.require_grid_space()

    # Remove tiny imaginary round-off in physical space
    psi['g'] = np.real(psi['g'])

    u_initial = d3.skew(d3.Gradient(psi)).evaluate()  # Velocity: (-psi_y, psi_x).
    u_initial.change_scales(1)

    u['g'][0]=u_initial['g'][0]
    u['g'][1]=u_initial['g'][1]
    target_KE_mean = 1
    u.require_grid_space()

    KE_local_sum = np.sum(
        0.5 * (np.abs(u['g'][0])**2 + np.abs(u['g'][1])**2)
    )
    N_local = u['g'][0].size

    KE_global_sum = dist.comm.allreduce(KE_local_sum)
    N_global = dist.comm.allreduce(N_local)

    KE_mean_now = KE_global_sum / N_global
    scale = np.sqrt(target_KE_mean / (KE_mean_now + 1.0e-300))

    u['g'][0] *= scale
    u['g'][1] *= scale

    logger.info(
        "Initial velocity rescaled: KE_mean_before=%.3e, target_KE_mean=%.3e, scale=%.3e"
        % (KE_mean_now, target_KE_mean, scale)
    )

if initial_condition == "shear_layer":
    # Initialize a periodic shear layer on y in [0, Ly].
    y_center = y - Ly/2
    u['g'][0] = 1/2 + 1/2 * (
        np.tanh((y_center - 0.5)/0.1)
        - np.tanh((y_center + 0.5)/0.1)
    )
    u['g'][1] = 0.0
    u['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(y_center - 0.5)**2/0.01)

logger.info(
    "Applied %s initial condition with Lx=%.3e, Ly=%.3e, Kmax=%i, Nx=%i, Ny=%i"
    % (initial_condition, Lx, Ly, Kmax, Nx, Ny)
)
# Check the field/operator and eight-sample reconstruction without advancing.
if args.check_only:
    probe = dist.VectorField(coords, name="check_probe", bases=(xbasis, ybasis))
    mx = Nx * 2.0 * np.pi / Lx
    my = Ny * 2.0 * np.pi / Ly
    coeffs, info = compute_formula32_coefficients(u, probe, mx, my)
    checks = []
    for tx, ty in ((0.173, -0.419), (-0.721, 0.613), (1.1, 0.3)):
        direct = epsilon_at_alpha(u, probe, tx/mx, ty/my)
        predicted = formula32_epsilon(coeffs, tx/mx, ty/my, mx, my)
        checks.append(abs(direct-predicted))
    scale = max(1.0, float(np.max(np.abs(coeffs))))
    if info["rank"] != 8 or max(checks) > 1e-9*scale:
        raise RuntimeError(f"MSM reconstruction check failed: {info}, {checks}")
    if dist.comm.rank == 0:
        print("PASS: initialized fields; full-rank eight-sample MSM; independent shift checks.")
    raise SystemExit(0)

if dist.comm.rank == 0:
    Path(outputPATH).mkdir(parents=True, exist_ok=False)
    (Path(outputPATH) / "configuration.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, indent=2) + "\n",
        encoding="utf-8",
    )
dist.comm.Barrier()
# Analysis
# snapshots are evaluated manually after any alias-control reverse shift.
snapshots = solver.evaluator.add_file_handler(f'{outputPATH}/snapshots', iter=10**18, max_writes=1)
snapshots.add_task(p, name='pressure')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(u, name='velocity')

snapshots.add_task(d3.div(u), name='divergence')
snapshots.add_task(0.5*(u@u), name='KE_density')
snapshots.add_task(alpha_x_output, name='alpha_x')
snapshots.add_task(alpha_y_output, name='alpha_y')

# Make the first output set index be 0 instead of the default 1.
snapshots.set_num = 0
snapshots.file_write_num = 0
snapshots.total_write_num = 0

# Force evaluation at the current solver state, i.e. initial condition.
alpha_x_output['g'] = 0.0
alpha_y_output['g'] = 0.0
solver.evaluator.evaluate_handlers(
    [snapshots],
    iteration=solver.iteration,
    sim_time=solver.sim_time,
    wall_time=0.0,
    timestep=0.0
)

logger.info("outputPATH=%s", outputPATH)
logger.info(
    "Initial diagnostics: KE=% .3e, convective_energy_rate=% .3e"
    % (
        evaluate_scalar(kinetic_energy_integral(u)),
        evaluate_scalar(total_convective_energy_rate(u)),
    )
)
logger.info("Initial condition written to snapshots_s0.h5")

# Mark the t=0 output as already handled.
snapshots.last_iter_div = 0
# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

u_probe = dist.VectorField(coords, name='u_probe', bases=(xbasis, ybasis))
M_alias_x = Nx * 2.0 * np.pi / Lx
M_alias_y = Ny * 2.0 * np.pi / Ly
alpha_x_prev = None
alpha_y_prev = None
last_alias_control = None

if use_formula32_alias_control and abs(dealias - 1.0) > 1e-14:
    logger.info(
        "Formula-(32) control is active with dealias=%.3e. "
        "For alias-dissipation control, use dealias=1 or an aliased sampling basis."
        % dealias
    )

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        applied_alpha = None

        if not np.isfinite(timestep):
            raise FloatingPointError(
                f"NaN/Inf timestep detected at iteration={solver.iteration}, "
                f"time={solver.sim_time}"
            )

        # Reconstruct and select the zero-power shift from the current velocity.
        if use_formula32_alias_control:
            coeffs, shift_info = compute_formula32_coefficients(
                u,
                u_probe,
                M_alias_x,
                M_alias_y,
            )
            alpha_x, alpha_y, predicted_eps, solve_info = solve_alphaxy_from_formula32(
                coeffs,
                M_alias_x,
                M_alias_y,
                eps_target,
                alpha_x_prev=alpha_x_prev,
                alpha_y_prev=alpha_y_prev,
                search_n=formula32_search_n,
            )
            actual_eps = epsilon_at_alpha(
                u,
                u_probe,
                alpha_x,
                alpha_y,
            )
            apply_spectral_translation_to_velocity(u, alpha_x, alpha_y)
            applied_alpha = (alpha_x, alpha_y)
            alpha_x_prev = alpha_x
            alpha_y_prev = alpha_y
            last_alias_control = {
                "X1": coeffs[0],
                "X2": coeffs[1],
                "X3": coeffs[2],
                "X4": coeffs[3],
                "Px": coeffs[4],
                "Qx": coeffs[5],
                "Py": coeffs[6],
                "Qy": coeffs[7],
                "q_x": shift_info["q_x"],
                "q_y": shift_info["q_y"],
                "fit_rank": shift_info["rank"],
                "fit_cond": shift_info["cond"],
                "fit_linf": shift_info["fit_linf"],
                "eps0": shift_info["eps0"],
                "alpha_x": alpha_x,
                "alpha_y": alpha_y,
                "predicted_eps": predicted_eps,
                "actual_eps": actual_eps,
                "eps_min": solve_info["eps_min"],
                "eps_max": solve_info["eps_max"],
                "eps_range": solve_info["eps_range"],
                "target_inside": solve_info["target_inside"],
                "best_error": solve_info["best_error"],
            }

        # Advance in the shifted frame, then restore coordinates before output.
        try:
            solver.step(timestep)
        finally:
            if applied_alpha is not None:
                apply_spectral_translation_to_velocity(
                    u,
                    -applied_alpha[0],
                    -applied_alpha[1],
                )

        # Stop immediately if the solution has crashed.
        u.require_grid_space()
        if not np.all(np.isfinite(u['g'])):
            raise FloatingPointError(
                f"NaN/Inf detected in u at iteration={solver.iteration}, "
                f"time={solver.sim_time}"
            )

        if solver.iteration % args.snapshot_every == 0:
            if last_alias_control is None:
                alpha_x_output['g'] = 0.0
                alpha_y_output['g'] = 0.0
            else:
                alpha_x_output['g'] = last_alias_control["alpha_x"]
                alpha_y_output['g'] = last_alias_control["alpha_y"]
            solver.evaluator.evaluate_handlers(
                [snapshots],
                iteration=solver.iteration,
                sim_time=solver.sim_time,
                wall_time=0.0,
                timestep=timestep,
            )

        if (solver.iteration-1) % 10 == 0:
            if last_alias_control is None:
                logger.info(
                    'Iteration=%i, Time=%.3e, dt=%.3e, KE=% .3e, convective_energy_rate=% .3e, outputPATH=%s'
                    % (
                        solver.iteration,
                        solver.sim_time,
                        timestep,
                        evaluate_scalar(kinetic_energy_integral(u)),
                        evaluate_scalar(total_convective_energy_rate(u)),
                        outputPATH,
                    )
                )
            else:
                logger.info(
                    'Iteration=%i, Time=%.3e, dt=%.3e, KE=% .3e, convective_energy_rate=% .3e, '
                    'alpha=% .3e, '
                    'eps_target=% .3e, eps_model=% .3e, '
                    'eps_actual_alpha=% .3e, '
                    'eps_min=% .3e, eps_max=% .3e, target_inside=%i'
                    % (
                        solver.iteration,
                        solver.sim_time,
                        timestep,
                        evaluate_scalar(kinetic_energy_integral(u)),
                        evaluate_scalar(total_convective_energy_rate(u)),
                        last_alias_control["alpha_x"],
                        eps_target,
                        last_alias_control["predicted_eps"],
                        last_alias_control["actual_eps"],
                        last_alias_control["eps_min"],
                        last_alias_control["eps_max"],
                        last_alias_control["target_inside"],
                    )
                )
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
