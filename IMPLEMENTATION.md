# Implementation and provenance

## One-dimensional implementation

The update functions in `burgers1d/solver.py` are extracted verbatim from
`burgers/lcs_burgers/BurgersSolver.py`. The first-order split update is retained:
convection, then molecular viscosity. The Smagorinsky reference includes its
original explicit SGS update between these two operations. No forcing is used.

The DSM functions in `burgers1d/control.py` are extracted verbatim from
`diagnostics/check_1d_inverse_shift_feasibility.py`, which uses the same normalized
Fourier coefficients and phase-control convention as the existing calculations.
Their hashes are listed in PROVENANCE.json and checked by `check_source.py`.

The Fourier convention is u_hat=FFT(u)/M and u=IFFT(u_hat)*M. The mean and
Nyquist modes are zero. The kinetic energy is K=0.5*sum_signed |u_hat|^2,
equivalently sum_positive |u_hat_k|^2 for a real velocity.

The DSM samples the instantaneous convective power at alpha=0 and
alpha=pi/(2*M). The non-aliased resolved contribution sums to zero, so this
power is the net aliasing power. A positive requested dissipation epsilon is
converted to target power -epsilon. The original two sine branches and
previous-phase continuity rule are retained. An unreachable target is clipped
to the instantaneous amplitude; no velocity rescaling is added.

`run_burgers.py` is a new portable input/output driver. It replaces absolute
paths and saved-data prerequisites with command-line options and generation of
the DNS restart/target. The default parameters follow the current small-grid
manuscript, not the separate large-grid exploratory studies. Its initial phases
use Python's random.Random(2), the original k_L=2 spectrum, and the original
real-field FFT/IFFT round trip.

The DNS-prescribed target is computed from successive filtered DNS energies:
epsilon_sgs[n]=-(K_filtered[n+1]-K_filtered[n])/dt-Dnu_filtered[n]. The original
forward difference is retained. Molecular viscosity is not counted twice.
For different target sampling intervals the continuation interpolates the target
in time; exact manuscript-step reproduction requires the documented common dt.

The timing functions are extracted from the original
`post_burger/efficiency_benchmark/benchmark_aliasing_efficiency.py`. Plotting
imports and plotting/export functions were omitted. The four numerical timing
kernels, random field generator and sampling procedure are preserved verbatim.

## Two-dimensional implementation

`ns2d/run_ns.py` is adapted from
`dedalus-master/NS_aliasing/essai_2D_aliasing/2D_period_energy_aliasing_zero_DNS_alphax_equal_y.py`.
The existing eight-sample basis, full-rank reconstruction, diagonal root search,
field translation, RK222 update and inverse translation are preserved. The
original random streamfunction construction and global energy normalization
are retained, including the original Nyquist treatment.

Packaging changes are limited to a command-line preamble, portable output path,
selection of MSM/reference/raw modes, parameter options, configuration recording,
and dry-run/check-only entries. The default stop time is 10 (the manuscript's
reported energy interval); the source script's current hard-coded stop time is
20. The default snapshot cadence remains 1000 iterations. No numerical tuning,
additional SGS term or branch-dependent shifts have been introduced.

The source uses functions named `epsilon_at_alpha` and `formula32_epsilon` to
evaluate the domain-integrated signed convective power, rather than the positive
mean dissipation. Their names and signs are preserved. All distributed MSM
cases have a zero target, so multiplying the entire scalar condition by -1 or
by the domain area does not change the selected zero. The package does not
expose a nonzero-target two-dimensional LES option.

## Checks and limits

`check_source.py` checks unchanged function text, real-field energy normalization,
FFT power against direct folded-triad summation, phase selection and saturation,
zero total dealiased convective power, the original split update, and the
eight-sample matrix. Dedalus `--check-only` additionally evaluates the actual
operator at shifts not used to construct the matrix, without advancing time.

The local check environment used Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1,
mpi4py 4.1.2, h5py 3.16.0, and Dedalus 3.0.5. The environment files describe
installable dependencies; they are not a claim that all compatible versions
produce bit-for-bit identical results.

This packaging task does not recompute the manuscript's full trajectories,
error tables or timings. The portable drivers preserve the identified kernels;
a new run is needed before claiming exact regeneration of every published curve.
