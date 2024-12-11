An algorithm called Newton-Schulz powers [Muon](https://github.com/KellerJordan/Muon), but...

1. PyTorch only captures 48% of the max throughput of an H100 running Newton-Schulz
2. Distributing across more GPUs with sharding costs data movement

Symmetric matrix multiply $A A^\top$ (symmul) is one step in Newton-Schulz.

We present a single-threaded speedy kernel that hits 167% H100 throughput on symmul (1650 TFLOPs).

How is that possible? Just do half the work: compute lower triangular blocks and copy them up.

Right now it's a bit buggy but works when `A = torch.ones(n, n)`.

As far as future directions, to make Muon work at industry scale, the important next step is building a distributed kernel across H100 nodes.

## How to run

Well... actually it's kind of hard to get ThunderKittens set up. But basically you need C++ 20, CUDA >12.3, and NVCC. Then to `make` the kernel, you need to first `source env.src` in your local copy of ThunderKittens. The Makefile in `symmul` creates a module called `symmul` that you can import in Python. Try it out with `python test_symmul.py`. You'll need an H100 to run things due to the WGMMAs.

