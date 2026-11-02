# DSM and MSM

Phase-shift control of aliasing dissipation in Fourier pseudospectral calculations.

This repository provides two implementations: the **Double-Shifting Method
(DSM)** for the one-dimensional Burgers equation and the **Multiple-Shifting
Method (MSM)** for the two-dimensional incompressible Navier--Stokes equations.
DSM includes zero-target and prescribed SGS-dissipation calculations. The
distributed MSM cases use a zero target.

## Get started

Clone the private repository using a GitHub account with access:

```bash
git clone https://github.com/bapzhao/DSM-MSM.git
cd DSM-MSM
```

For the one-dimensional calculations:

```bash
conda env create -f environment-1d.yml
conda activate 1dburgers
python check_source.py
python burgers1d/run_burgers.py --case zero-dsm --cutoff 32 --stop 0.002 --output outputs/1d/example
```

For the two-dimensional calculations:

```bash
conda env create -f environment-2d.yml
conda activate nsdedalus
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python ns2d/run_ns.py --method msm --kmax 16 --check-only
```

The small example and initialization check are intended to check installation.
The manuscript configurations and execution order are documented in
[Usage](docs/USAGE.md).

## Source and examples

| Directory or file | Contents |
| --- | --- |
| [burgers1d/](burgers1d/) | Burgers update, DSM control, calculation entry and timing comparison |
| [ns2d/](ns2d/) | Dedalus MSM, dealiased reference and uncontrolled calculation |
| [examples/1d/](examples/1d/) | Sequential reference and LES launch script |
| [examples/2d/](examples/2d/) | Serial or MPI launch script |
| [docs/USAGE.md](docs/USAGE.md) | Installation, parameters, commands and output conventions |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Numerical conventions and packaging changes |
| [check_source.py](check_source.py) | Source consistency and small numerical checks |
| [PROVENANCE.json](PROVENANCE.json) | Original source and extracted function hashes |

Simulation results, saved initial fields, plots and manuscript files are not
included. The calculation programs generate their required DNS restart and
target-history inputs. Output directories are selected explicitly and existing
directories are not overwritten.

## Checks

```bash
python check_source.py
python burgers1d/run_burgers.py --case dns-reference --dry-run
python ns2d/run_ns.py --method msm --dry-run
```

The checks cover the unchanged source functions, energy normalization, direct
folded-triad and FFT power agreement, phase selection, saturation, the split
update and the MSM sampling matrix. With Dedalus installed, `--check-only`
also checks the actual operator at independent shifts.

An optional [GitHub Actions template](docs/source-checks.yml.example) runs the
NumPy source and numerical-identity checks. It can be enabled by placing it at
`.github/workflows/source-checks.yml` with an account authorized to manage workflows.
The local Dedalus serial and MPI checks are described in [Implementation](IMPLEMENTATION.md).
These checks do not establish full reproduction of every manuscript trajectory.

## Manuscript and citation

This software accompanies the manuscript by Yicong Zhao, Chensheng Luo,
Jian Fang, Liang Shao and Le Fang, currently titled
*Using aliasing error as numerical dissipation in implicit large-eddy simulation
of the one-dimensional Burgers equation*. The manuscript is under revision;
publication details and an archive DOI will be added when available.

## Availability and license

The private source repository for the CPC revision is
[bapzhao/DSM-MSM](https://github.com/bapzhao/DSM-MSM).
A distribution license and a public archived version have not yet been assigned.
See [NOTICE.md](NOTICE.md). External dependencies retain their own licenses.

The repository organization draws on
[Xcompact3d](https://github.com/Xcompact3d/Incompact3d) and
[NNPred](https://github.com/Weishuo93/NN_Pred): calculation source,
usage documentation, examples and checks are provided separately.
No source code from those projects is included.
