# triton-kernels

A collection of GPU kernels written in [Triton](https://github.com/triton-lang/triton).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install triton torch
```

## Layout

```
kernels/   # Triton kernel implementations
tests/     # Correctness and benchmark tests
```
