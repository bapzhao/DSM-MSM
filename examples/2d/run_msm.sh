#!/usr/bin/env bash
# Run the existing two-dimensional zero-target MSM case.
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: bash examples/2d/run_msm.sh OUTPUT_DIRECTORY [MPI_PROCESSES]" >&2
    exit 2
fi
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON:-python}"
processes="${2:-1}"
if ! [[ "$processes" =~ ^[1-9][0-9]*$ ]]; then
    echo "MPI_PROCESSES must be a positive integer." >&2
    exit 2
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [ "$processes" -eq 1 ]; then
    "$python_bin" "$source_root/msm/run_ns.py" --method msm --output "$1"
else
    mpiexec -n "$processes" "$python_bin" "$source_root/msm/run_ns.py" --method msm --output "$1"
fi
