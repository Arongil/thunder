import symmul
import matplotlib.pyplot as plt
import numpy as np
import torch

assert torch.cuda.is_available()
device = "cuda"

n = 8192
dtype = torch.bfloat16
A = torch.ones((n, n), dtype=dtype, device=device)
B = A.T.contiguous()
C = torch.zeros((n, n), dtype=dtype, device=device)

def plot_correctness_heatmap(C):
    # Convert to CPU numpy array for plotting
    C_np = C.to(torch.float32).cpu().numpy()
    # Create boolean mask where values equal 4096
    correct_mask = (C_np == 4096)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(correct_mask, cmap='RdYlGn')
    plt.colorbar(label='Correct (4096)')
    plt.title('Output Correctness Heatmap')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index') 
    plt.savefig('correctness.png', dpi=600)

def benchmark(f, A, B, C, warmup=3, plot=False):
    """Benchmark a matrix multiplication function f(A,B,C)"""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup runs
    for _ in range(warmup):
        f(A, B, C)
        torch.cuda.synchronize() # Make sure previous run is done
        C.zero_() # Zero out C without caching

    # Timed run
    start.record()
    f(A, B, C)
    end.record()

    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    flops = 2 * A.shape[0] * A.shape[1] * B.shape[1]  # multiply-add is 2 operations
    tflops = (flops / (elapsed_time / 1000)) / 1e12  # convert ms to s and flops to tflops
    print(f"{f.__name__} execution time: {elapsed_time:.2f} ms ({tflops:.2f} TFLOPs/s)")

    if plot:
        plot_correctness_heatmap(C)

    assert torch.allclose(C, A @ B)

benchmark(symmul.matmul, A, B, C, warmup=9, plot=False)
benchmark(symmul.symmul, A, B, C, warmup=9, plot=True)
