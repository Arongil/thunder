import symmul
import torch

assert torch.cuda.is_available()
device = "cuda"

n = 4096
dtype = torch.bfloat16
A = torch.ones((n, n), dtype=dtype, device=device)
B = A.T.contiguous()
C = torch.zeros((n, n), dtype=dtype, device=device)

def benchmark(f, A, B, C):
    """Benchmark a matrix multiplication function f(A,B,C)"""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup runs
    f(A, B, C)
    f(A, B, C) 
    f(A, B, C)
    C = C * 0

    # Timed run
    start.record()
    f(A, B, C)
    end.record()

    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    flops = 2 * A.shape[0] * A.shape[1] * B.shape[1]  # multiply-add is 2 operations
    tflops = (flops / (elapsed_time / 1000)) / 1e12  # convert ms to s and flops to tflops
    print(f"{f.__name__} execution time: {elapsed_time:.2f} ms ({tflops:.2f} TFLOPs/s)")

    assert torch.allclose(C, A @ B)

benchmark(symmul.symmul4096_4096, A, B, C)
benchmark(symmul.matmul4096_4096, A, B, C)