import torch
import time
import functools
import ctypes

torch.backends.cudnn.benchmark = True

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

### Symmetric-general matrix multiplication (sygm)

def symg(X, Y):
    return X @ Y.T

symg_compiled = torch.compile(symg)

def symg_data_generator(dim, device, dtype=torch.bfloat16):
    return (torch.randn(size=dim, device=device, dtype=dtype), torch.randn(size=dim, device=device, dtype=dtype))

### Symmetric rank k (syrk)

# Load CUDA runtime and cuBLAS libraries
cuda = ctypes.CDLL('libcudart.so')
cublas = ctypes.CDLL('libcublas.so')

# Define the cuBLAS handle type and operation types
cublasHandle_t = ctypes.c_void_p
CUBLAS_FILL_MODE_LOWER = 1
CUBLAS_OP_N = 0

def create_cublas_handle():
    handle = cublasHandle_t()
    status = cublas.cublasCreate_v2(ctypes.byref(handle))
    if status != 0:
        raise RuntimeError(f"cuBLAS initialization failed with status {status}")
    return handle

def destroy_cublas_handle(handle):
    if handle:
        cublas.cublasDestroy_v2(handle)

def syrk(X):
    """Compute X @ X.T using cuBLAS SYRK"""
    # Convert to column-major format for cuBLAS
    X_t = X.t().contiguous()

    # Get dimensions
    n = X.size(0)
    k = X.size(1)

    # Initialize output
    C = torch.zeros(n, n, device=X.device, dtype=X.dtype)

    # Set scalar parameters
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    # Get pointers to GPU memory
    X_ptr = ctypes.c_void_p(X_t.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Initialize cuBLAS
    handle = create_cublas_handle()

    try:
        status = cublas.cublasSsyrk_v2(
            handle,
            CUBLAS_FILL_MODE_LOWER,  # Fill lower triangle
            CUBLAS_OP_N,             # No transpose (we pre-transposed)
            n,                       # rows/cols in C
            k,                       # cols in original X
            ctypes.byref(alpha),
            X_ptr,                   # Input matrix (transposed)
            n,                       # lda: leading dimension of X must be ≥ n
            ctypes.byref(beta),
            C_ptr,
            n                        # ldc: leading dimension of C
        )

        if status != 0:
            raise RuntimeError(f"cuBLAS SYRK failed with status {status}")

        # Fill upper triangle by reflecting lower triangle
        C = C + torch.tril(C, -1).t()

        return C

    finally:
        destroy_cublas_handle(handle)

syrk_compiled = torch.compile(syrk)

def syrk_data_generator(dim, device, dtype=torch.bfloat16):
    return (torch.randn(size=dim, device=device, dtype=dtype),)

def test_syrk_correctness():
    """Verify SYRK implementation against various test cases"""
    device = torch.device("cuda")
    test_cases = [
        # Identity matrix
        torch.eye(2, device=device),
        # 2x1 matrix
        torch.tensor([[1.], [2.]], device=device),
        # 2x2 simple matrix
        torch.tensor([[1., 2.], [3., 4.]], device=device),
        # 2x3 matrix
        torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=device),
        # Matrix of ones
        torch.ones(2, 2, device=device),
        # Random larger matrices
        torch.randn(16, 32, device=device),
        torch.randn(64, 16, device=device),
    ]

    #print("\nRunning SYRK correctness tests...")
    all_passed = True

    for i, X in enumerate(test_cases):
        # Compute with our SYRK implementation
        result = syrk(X)
        # Compute ground truth with PyTorch
        expected = X @ X.t()

        # Check if results match
        matches = torch.allclose(result, expected, rtol=1e-5, atol=1e-5)
        all_passed &= matches

        if not matches:
            print(f"Test case {i+1}: {'✓' if matches else '✗'}")
            print(f"Input shape: {X.shape}")
            print("Absolute difference:")
            print((result - expected).abs().max().item())

    #print(f"\nOverall test result: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed

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
        "Newton-Schulz":                     (False, ns, ns_data_generator, 256),
        "Compiled Newton-Schulz":            (False, ns_compiled, ns_data_generator, 256),
        "Stripped Newton-Schulz":            (False, sns, sns_data_generator, 256),
        "Compiled Stripped Newton-Schulz":   (False, sns_compiled, sns_data_generator, 256),
        "Symmetric Matmul":                  (False, symm, symm_data_generator, 2048),
        "Compiled Symmetric Matmul":         (True, symm_compiled, symm_data_generator, 2048),
        "Symmetric-General Matmul":          (False, symg, symg_data_generator, 2048),
        "Compiled Symmetric-General Matmul": (False, symg_compiled, symg_data_generator, 2048),
        "SYRK":                              (True, syrk, syrk_data_generator, 2048),
        "Compiled SYRK":                     (True, syrk_compiled, syrk_data_generator, 2048),
    }

    dims = [
        (16, 16),
        (256, 256),
        (256, 1024),
        (1024, 1024),
        (1024, 4096),
        (4096, 4096),
        #(4096, 16384),
        #(8192, 32768),
    ]

    check_correctness = True
    if check_correctness:
        if not test_syrk_correctness():
            print("\nSkipping benchmarks due to correctness failure")
            exit()

    for name, (do_benchmark, f, data_generator, iters) in benchmarks.items():
        if do_benchmark:
            pretty_print_benchmark(name, f, data_generator, dims, device, iters)

