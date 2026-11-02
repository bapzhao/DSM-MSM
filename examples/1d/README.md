# One-dimensional examples

Activate the `1dburgers` environment and run commands from the repository root.

A small installation check is:

```bash
python burgers1d/run_burgers.py --case zero-dsm --cutoff 32 --stop 0.002 --output outputs/1d/example
```

To execute the manuscript LES configurations in dependency order:

```bash
bash examples/1d/run_les.sh outputs/1d/les
```

The script generates the reference DNS restart and forward-difference SGS
target before starting DSM and comparison calculations. It does not fit model
coefficients or run the separate large-grid exploratory cases. Existing output
directories are not overwritten.

To use a specific interpreter, set `PYTHON` before launching the script.
Individual commands and the zero-target comparison are in
[Usage](../../docs/USAGE.md).
