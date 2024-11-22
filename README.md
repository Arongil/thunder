Speedy kernels for spectral dualization

An algorithm called Newton-Schulz powers [Muon](https://github.com/KellerJordan/Muon), but...

1. PyTorch only captures 26% of the max throughput of an H100 running Newton-Schulz
2. Distributing across more GPUs with sharding costs data movement

The repo makes a single-threaded speedy kernel that hits __% H100 throughput and a distributed kernel for 8xH100 nodes.


