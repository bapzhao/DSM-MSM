# DSM-MSM

Calculation code for the Double-Shifting Method (DSM, one-dimensional Burgers)
and Multiple-Shifting Method (MSM, two-dimensional incompressible Navier--Stokes).

Repository: https://github.com/bapzhao/DSM-MSM

## Calculation files

| File | Purpose |
| --- | --- |
| `dsm/run_burgers.py` | Run the one-dimensional DSM, DNS, and LES comparisons. |
| `dsm/control.py` | Evaluate aliasing power and select the DSM phase shift. |
| `dsm/solver.py` | Supply Fourier wavenumbers and Burgers time-stepping kernels. |
| `msm/run_ns.py` | Run the two-dimensional zero-target MSM and reference cases. |

Run commands below from the repository root. Each Python file starts with
usage notes; the two calculation entries also accept `--help`.

## Installation

Download the calculation files:

```bash
git clone https://github.com/bapzhao/DSM-MSM.git
cd DSM-MSM
```

Create the calculation environments from conda-forge:

```bash
conda env create -f environment-1d.yml
conda env create -f environment-2d.yml
```

## DSM

Activate the one-dimensional environment and generate the DNS reference:

```bash
conda activate 1dburgers
python dsm/run_burgers.py --case dns-reference --output outputs/reference
```

This generates the restart and target history used by the LES commands:

```bash
python dsm/run_burgers.py --case dsm-smag --cs 0.20 --restart outputs/reference/state_t0.300000.npz --output outputs/dsm020
python dsm/run_burgers.py --case dsm-smag --cs 0.27 --restart outputs/reference/state_t0.300000.npz --output outputs/dsm027
python dsm/run_burgers.py --case dsm-filtered --restart outputs/reference/state_t0.300000.npz --target outputs/reference/sgs_target.csv --output outputs/dsm_filtered
python dsm/run_burgers.py --case smag --cs 0.20 --restart outputs/reference/state_t0.300000.npz --output outputs/smag
python dsm/run_burgers.py --case coarse --restart outputs/reference/state_t0.300000.npz --output outputs/coarse
```

Zero-target comparison:

```bash
python dsm/run_burgers.py --case zero-dsm --output outputs/zero_dsm
python dsm/run_burgers.py --case zero-reference --output outputs/zero_reference
python dsm/run_burgers.py --case zero-raw --output outputs/zero_raw
```

## MSM

The distributed two-dimensional cases use a zero target.

```bash
conda activate nsdedalus
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python msm/run_ns.py --method msm --output outputs/msm
python msm/run_ns.py --method reference --output outputs/msm_reference
python msm/run_ns.py --method raw --output outputs/msm_raw
```

For MPI, prepend the launcher from the same environment, such as `mpiexec -n 4`.

Default parameters are set in the calculation entries; use `--help` for options.
Output directories must not already exist. The uncontrolled cases may become
unstable and stop. No simulation results or plotting programs are included.

No software license has been assigned.
