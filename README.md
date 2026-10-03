# triton-kernels

GPU kernels written in [Triton](https://github.com/triton-lang/triton), each with a
correctness test and a benchmark against its PyTorch baseline.

Benchmarks below were measured on a single **NVIDIA RTX 6000 Ada** (48 GB).

## Kernels

| Kernel | Op | Source | Test | Benchmark |
|--------|----|--------|------|-----------|
| `vector_add` | `x + y` | [vector_add.py](kernels/vector_add.py) | [test](tests/test_vector_add.py) | [bench](benchmarks/bench_vector_add.py) |
| `vector_subtract` | `x - y` | [vector_subtract.py](kernels/vector_subtract.py) | [test](tests/test_vector_subtract.py) | [bench](benchmarks/bench_vector_subtract.py) |

### `vector_add` — element-wise `x + y`

Both Triton and PyTorch are memory-bandwidth-bound and plateau around ~815 GB/s —
as expected for an op that does one add per two loads and a store.

![vector_add benchmark](docs/vector-add-performance.png)

### `vector_subtract` — element-wise `x - y`

Same bandwidth-bound profile as add; the Triton kernel matches the PyTorch baseline.

![vector_subtract benchmark](docs/vector-subtract-performance.png)

## Layout

```
kernels/              # kernel implementations (one module per kernel)
tests/                # pytest correctness tests (auto-skip without a GPU)
benchmarks/           # benchmarks vs. PyTorch; plots -> benchmark_outputs/
docs/                 # plots shown in this README
```

## Usage

Triton ships Linux + GPU wheels only, so kernels run on the GPU server rather than
locally. `sync.sh` handles the round trip (files sync to the login node over shared
NFS; commands run on the compute node via SSH `ProxyJump`):

```bash
./sync.sh push                                    # copy code to the server
./sync.sh gpus                                    # check which GPUs are free
./sync.sh run pytest -q                           # run the tests
./sync.sh run python benchmarks/bench_vector_add.py   # run a benchmark
./sync.sh pull                                    # bring plots/results back
```

Benchmarks auto-save a `.png` and `.csv` into `benchmark_outputs/`.

First-time server setup is one command (`./setup.sh`): it builds the
`triton-kernels` conda env from `environment.yml`, installs Triton, and runs the
tests. See [setup.sh](setup.sh) and [sync.sh](sync.sh) for details.
