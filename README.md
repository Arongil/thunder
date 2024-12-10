An algorithm called Newton-Schulz powers [Muon](https://github.com/KellerJordan/Muon), but...

1. PyTorch only captures 48% of the max throughput of an H100 running Newton-Schulz
2. Distributing across more GPUs with sharding costs data movement

The repo makes a single-threaded speedy kernel that hits 167% H100 throughput on the symmetric matrix multiply $A A^\top$ that is a substep of Newton-Schulz.

How is that possible? Just do half the work: compute lower triangular blocks and copy them up. The output is symmetric anyway.

Our kernel hits 1650 TFLOPs for symmetric matrix multiplication.

Right now it's a bit buggy but works when `A = torch.ones(n, n)`.

As far as future directions, to make Muon work at industry scale, the important next step is building a distributed kernel across H100 nodes.


