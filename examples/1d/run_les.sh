#!/usr/bin/env bash
# Generate the DNS input, then run the existing one-dimensional LES cases.
set -euo pipefail
if [ "$#" -ne 1 ]; then
    echo "Usage: bash examples/1d/run_les.sh OUTPUT_DIRECTORY" >&2
    exit 2
fi
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON:-python}"
output_root="$1"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"$python_bin" "$source_root/dsm/run_burgers.py" --case dns-reference --output "$output_root/reference"
restart="$output_root/reference/state_t0.300000.npz"
"$python_bin" "$source_root/dsm/run_burgers.py" --case dsm-smag --cs 0.20 --restart "$restart" --output "$output_root/dsm_cs020"
"$python_bin" "$source_root/dsm/run_burgers.py" --case dsm-smag --cs 0.27 --restart "$restart" --output "$output_root/dsm_cs027"
"$python_bin" "$source_root/dsm/run_burgers.py" --case dsm-filtered --restart "$restart" --target "$output_root/reference/sgs_target.csv" --output "$output_root/dsm_filtered"
"$python_bin" "$source_root/dsm/run_burgers.py" --case smag --cs 0.20 --restart "$restart" --output "$output_root/smag_cs020"
"$python_bin" "$source_root/dsm/run_burgers.py" --case coarse --restart "$restart" --output "$output_root/coarse"
