# Two-dimensional examples

Activate the `nsdedalus` environment. Use its Python and MPI launcher together.

Initialize a small field and check the eight-sample reconstruction without
advancing time or writing results:

```bash
python msm/run_ns.py --method msm --kmax 16 --check-only
```

Run the manuscript MSM configuration in serial:

```bash
bash examples/2d/run_msm.sh outputs/2d/msm
```

Or use four MPI processes:

```bash
bash examples/2d/run_msm.sh outputs/2d/msm_mpi4 4
```

The default configuration is Kmax=128, viscosity 7.23e-4, zero target, RK222,
maximum time step 1e-4 and final time 10. Reference and raw calculations use
the same entry with `--method reference` and `--method raw`, respectively.
Parameter and output details are in [Usage](../../docs/USAGE.md).
