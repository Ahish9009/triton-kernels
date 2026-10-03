# triton-kernels

A collection of GPU kernels written in [Triton](https://github.com/triton-lang/triton).

## Layout

```
kernels/      # Triton kernel implementations (one module per kernel)
tests/        # Correctness tests (pytest; auto-skip without a CUDA GPU)
benchmarks/   # Performance benchmarks vs. PyTorch baselines
```

Each kernel follows the same pattern: the implementation lives in `kernels/`,
a correctness test in `tests/`, and a benchmark in `benchmarks/`. See
`vector_add` for the reference example.

## Development (this machine)

Triton ships **Linux + GPU wheels only** — there is no macOS build, so it
can't be installed or run locally on a Mac. Develop here, run on the GPU
server (below).

A conda env with Python, PyTorch, and pytest is set up for editing and
linting:

```bash
conda activate triton-kernels
```

## Running on the remote GPU server

On the CUDA machine, create the env and install the full toolchain:

```bash
conda create -y -n triton-kernels python=3.11
conda activate triton-kernels
pip install -r requirements.txt
pip install triton        # Linux + GPU only
```

Then, from the repo root:

```bash
pytest                              # run correctness tests
python benchmarks/bench_vector_add.py   # run a benchmark
```
