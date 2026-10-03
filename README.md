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

Sync the repo to the server and set up the env — one command each:

```bash
./sync.sh push     # mirror this repo to ahishd@trinity.vision.cs.cmu.edu
```

Then, on the server, from the repo root, run the setup script once:

```bash
./setup.sh         # creates the conda env, installs Triton, runs the tests
```

`setup.sh` builds the `triton-kernels` conda env from `environment.yml`,
installs Triton (GPU-only, not in the portable env file), verifies the
install, and runs the suite.

### Day-to-day workflow

```bash
./sync.sh push                 # push local edits to the server (shared NFS: visible on all nodes)
./sync.sh gpus                 # check which GPUs are free before launching
./sync.sh run                  # run `pytest -q` on the server in the env
./sync.sh run python benchmarks/bench_vector_add.py   # or any command
./sync.sh pull                 # bring results/artifacts back locally
```

Files sync to the **login node** (`trinity.vision.cs.cmu.edu`); thanks to
shared NFS they're then visible on every node. Commands (`run`/`gpus`) execute
on the **compute node** (`trinity-0-3`), reached automatically by jumping
through the login node (SSH `ProxyJump`). Override either target:

```bash
REMOTE_HOST=ahishd@login   COMPUTE_HOST=ahishd@trinity-0-5   ./sync.sh run pytest -q
```

> Tip: `ssh-copy-id ahishd@trinity.vision.cs.cmu.edu` once gives passwordless
> sync (and passwordless jumps to the compute node, since the login node is the
> jump host).
