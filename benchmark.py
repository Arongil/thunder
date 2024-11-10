import torch
import time
import functools

#####################################
## Kernels

### Newton-Schulz

def ns(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

ns_compiled = torch.compile(ns)

def ns_data_generator(dim, device, dtype=torch.bfloat16):
    return (torch.randn(size=dim, device=device, dtype=dtype), 5, 1e-7)

### Stripped Newton-Schulz

def sns(X):
    if X.size(0) > X.size(1):
        X = X.T
    for _ in range(5):
        A = X @ X.T
        B = A @ X
        X = 2 * X + 3 * B + 5 * A @ B
    return X

sns_compiled = torch.compile(sns)

def sns_data_generator(dim, device, dtype=torch.bfloat16):
    return (torch.randn(size=dim, device=device, dtype=dtype),)

### Symmetric matrix multiplication (symm)

def symm(X):
    if X.size(0) > X.size(1):
        X = X.T
    return X @ X.T

symm_compiled = torch.compile(symm)

def symm_data_generator(dim, device, dtype=torch.bfloat16):
    return (torch.randn(size=dim, device=device, dtype=dtype),)

#####################################
## Benchmarking

def benchmark(f, data_generator, iters=256):
    # Takes function f with arguments compatible with *data_generator()
    # 
    # Ex: f = lambda X, Y: X @ Y,
    #     data_generator = lambda: (torch.randn(64, 256), torch.randn(256, 64))
    # 
    # Returns the time f takes to run

    torch.cuda.synchronize()
    for _ in range(16):
        f(*data_generator())

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        f(*data_generator())
    t1 = time.time()

    f_and_data_generator_time = t1 - t0

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        data_generator()
    t1 = time.time()

    data_generator_time = t1 - t0

    f_time = f_and_data_generator_time - data_generator_time

    return f_time / iters

def pretty_print_benchmark(name, f, data_generator, dims, device, iters):
    print(f"{name:^41}")
    print("-"*41)
    print(f"{'dim':^20}|{'time (ms)':^20}")
    for dim in dims:
        gen = functools.partial(data_generator, dim=dim, device=device)
        sec = benchmark(f, gen, iters=iters)
        print(f"{str(dim):^20}|{sec*1000:^20.3f}")
    print("")

if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    benchmarks = {
        "Newton-Schulz":                   (False, ns, ns_data_generator, 256),
        "Compiled Newton-Schulz":          (False, ns_compiled, ns_data_generator, 256),
        "Stripped Newton-Schulz":          (False, sns, sns_data_generator, 256),
        "Compiled Stripped Newton-Schulz": (False, sns_compiled, sns_data_generator, 256),
        "Symmetric Matmul":                (True, symm, symm_data_generator, 2048),
        "Compiled Symmetric Matmul":       (True, symm_compiled, symm_data_generator, 2048),
    }

    dims = [
        (16, 16),
        (256, 256),
        (256, 1024),
        (1024, 1024),
        (1024, 4096),
        (4096, 4096),
        (4096, 16384),
        (8192, 32768),
    ]

    for name, (do_benchmark, f, data_generator, iters) in benchmarks.items():
        if do_benchmark:
            pretty_print_benchmark(name, f, data_generator, dims, device, iters)

