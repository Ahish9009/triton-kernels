# triton-kernels

GPU kernels written in [Triton](https://github.com/triton-lang/triton), each with a
correctness test and a benchmark against its PyTorch baseline.

Benchmarks below were measured on a single **NVIDIA RTX 6000 Ada** (48 GB).

## Kernels

| Kernel | Op | Source | Test | Benchmark |
|--------|----|--------|------|-----------|
| `vector_add` | `x + y` | [vector_add.py](kernels/vector_add.py) | [test](tests/test_vector_add.py) | [bench](benchmarks/bench_vector_add.py) |
| `vector_subtract` | `x - y` | [vector_subtract.py](kernels/vector_subtract.py) | [test](tests/test_vector_subtract.py) | [bench](benchmarks/bench_vector_subtract.py) |
| `matmul_naive` | `A @ B` | [matmul_naive.py](kernels/matmul_naive.py) | [test](tests/test_matmul_naive.py) | [bench](benchmarks/bench_matmul_naive.py) |
| `matmul` | `A @ B` (tiled) | [matmul.py](kernels/matmul.py) | [test](tests/test_matmul.py) | [bench](benchmarks/bench_matmul.py) |

### `vector_add` — element-wise `x + y`

Both Triton and PyTorch are memory-bandwidth-bound and plateau around ~815 GB/s —
as expected for an op that does one add per two loads and a store.

![vector_add benchmark](docs/vector-add-performance.png)

### `vector_subtract` — element-wise `x - y`

Same bandwidth-bound profile as add; the Triton kernel matches the PyTorch baseline.

![vector_subtract benchmark](docs/vector-subtract-performance.png)

### `matmul_naive` — matrix multiply `A @ B`

A straightforward first-pass matmul: one program instance per output element,
each looping over the shared inner dimension in `BLOCK_SIZE` chunks and
accumulating the dot product. It's correctness-first and not tiled/blocked for
data reuse, so it trails PyTorch's cuBLAS-backed `@` — the plot shows the gap a
naive implementation leaves on the table (and the baseline to optimize against).

![matmul_naive benchmark](docs/matmul_naive-performance.png)

### `matmul` — tiled matrix multiply `A @ B`

A proper tiled implementation: each program computes a `BLOCK_SIZE_Y × BLOCK_SIZE_X`
output tile, looping over the inner dimension in `BLOCK_SIZE_K` chunks and
accumulating in registers with `tl.dot` (TF32 tensor cores). Because every loaded
tile is reused across the whole output block, it's compute-bound rather than
bandwidth-bound.

Block-size tuning trades performance across the size range — no single tile shape
is best everywhere. The config that tracks PyTorch's cuBLAS `@` best at the mid
sizes isn't the one that tracks best at the largest matrices:

| Block config A | Block config B (better at large sizes) |
|:---:|:---:|
| ![matmul config A](docs/matmul-blocks-a.png) | ![matmul config B](docs/matmul-blocks-b.png) |

The kernel now uses `@triton.autotune`, which benchmarks a set of block-size /
`num_stages` / `num_warps` configs per problem shape and caches the fastest. That
lets it pick the smaller tiles at mid sizes (enough program instances to fill the
SMs) and the larger tiles at big sizes (more data reuse) automatically. The result
tracks cuBLAS across the whole range and edges *above* it around 1024:

![autotuned matmul](docs/matmul-blocks-c.png)

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
