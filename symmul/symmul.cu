#include "kittens.cuh"
#include "prototype.cuh"
//#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;  // shared tile of size 64 by 64
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };  // outer product input
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };     // outer product result
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; }; // register_tile of size 16 by N_BLOCK * however many columns we're doing
};
template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
      // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        // This is finding the number of "super" big blocks it can calculate in parallel,
        // then doing the remaining rows one-by-one
        int Rblocks = args.globals.C.rows / (M_BLOCK*64), Cblocks = args.globals.C.cols / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,
                           (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AB(
                args.state.accum, // dest registers
                args.input.a[warpgroup::groupid()], // A matrix
                reinterpret_cast<wide_tile&>(args.input.b) // B matrix
            );
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::warpid() == 0) for(int i = 0; i < N_BLOCK; i++) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                             {args.common.coord.x, args.common.coord.y+i});
                tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
            }
            zero(args.state.accum);
            if(laneid() == 0) arrive(args.finish_finished);
		    args.globals.C[{0, 0, 0, 0}] = args.globals.A[{0,0,0,0}];
        }
    };
};

struct copy_layout {
    using base_tile     = st_bf<64, 64>;  // shared tile of size 64 by 64
    using global_layout = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals { global_layout A, B, C; };
    struct input_block { base_tile a[1]; };
    struct common_state   { int2 coord; };
};
struct copy_template {
    using layout = copy_layout;
    static constexpr int NUM_CONSUMER_WARPS=4, INPUT_PIPE_STAGES=1, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(1, 1, 4096/64);
    }
    // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        args.num_iters = args.globals.A.cols/64; 
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
        }
        __device__ static void load(producer_load_args<layout> args) {
            /*
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                tma::load_async(args.input.a[0], args.globals.A, {0, args.iter}, args.inputs_arrived);
            }
            */
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            /*
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
            */
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            //warpgroup::store(args.globals.C[warpgroup::groupid()], args.finish.a[warpgroup::groupid()]);
            //if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};

constexpr bool NCU = false;
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>

void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

template<typename mmt>
void matmul(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using global_layout = typename mmt::layout::global_layout;
    using globals  = typename mmt::layout::globals;
    global_layout Ag{d_A, nullptr, nullptr, M, K};
    global_layout Bg{d_B, nullptr, nullptr, K, N};
    global_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

//using mmt = typename ::matmul_template<2,4,8>;
using mmt = typename ::copy_template;
using global_layout = typename mmt::layout::global_layout;
using globals = typename mmt::layout::globals;

/*
__global__ void one_kernel(const globals g) {
    if (blockIdx.x < g.C.batch && blockIdx.y < g.C.depth && blockIdx.z < g.C.rows && threadIdx.x < g.C.cols)
	    g.C[{blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x}] = g.A[{blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x}];
}
*/

void symmul4096_4096(globals g) {
	const size_t N = 4096;
	const size_t K = 4096;
    dim3 grid(mmt::grid(N, N, K));
    dim3 block(prototype::detail::NUM_THREADS_v<mmt>);

    // Create the layouts with explicit batch and depth values
    /*
    global_layout Ag{g.A.raw_ptr, nullptr, nullptr, N, K};  // batch=1, depth=1, rows=N, cols=K
    global_layout Bg{g.B.raw_ptr, nullptr, nullptr, K, N};  // batch=1, depth=1, rows=K, cols=N
    global_layout Cg{g.C.raw_ptr, nullptr, nullptr, N, N};  // batch=1, depth=1, rows=N, cols=N

    globals G{Ag, Bg, Cg};

    static_assert(Ag.batch == 1 && Ag.depth == 1);
    static_assert(Bg.batch == 1 && Bg.depth == 1);
    static_assert(Cg.batch == 1 && Cg.depth == 1);
    assert(Ag.batch == 1 && Ag.depth == 1);
    assert(Bg.batch == 1 && Bg.depth == 1);
    assert(Cg.batch == 1 && Cg.depth == 1);
    */

    // Print detailed information about the input matrices
    printf("\n=== Matrix Properties ===\n");
    printf("Matrix A:\n");
    printf("  Raw pointer: %p\n", g.A.raw_ptr);
    printf("  Batch: %d\n", g.A.batch);
    printf("  Depth: %d\n", g.A.depth); 
    printf("  Rows: %zu\n", g.A.rows);
    printf("  Cols: %zu\n", g.A.cols);

    printf("\nMatrix B:\n");
    printf("  Raw pointer: %p\n", g.B.raw_ptr);
    printf("  Batch: %d\n", g.B.batch);
    printf("  Depth: %d\n", g.B.depth);
    printf("  Rows: %zu\n", g.B.rows);
    printf("  Cols: %zu\n", g.B.cols);

    printf("\nMatrix C:\n"); 
    printf("  Raw pointer: %p\n", g.C.raw_ptr);
    printf("  Batch: %d\n", g.C.batch);
    printf("  Depth: %d\n", g.C.depth);
    printf("  Rows: %zu\n", g.C.rows);
    printf("  Cols: %zu\n", g.C.cols);

    printf("\nGrid dimensions: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    printf("Block dimensions: (%d, %d, %d)\n", block.x, block.y, block.z);
    printf("Shared memory size: %d\n", MAX_SHARED_MEMORY-1024);
    printf("======================\n\n");

    //one_kernel<<<grid, block>>>(G);
    //prototype::lcf::kernel<matmul_template<2,4,8>><<<grid, block, MAX_SHARED_MEMORY-1024>>>(g);
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(g);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Wait for kernel to finish and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    printf("Kernel completed successfully\n");
}

/*
void symmul4096_16384(globals g) {
	dim3 grid(mmt::grid(4096, 4096, 16384));
	dim3 block(prototype::detail::NUM_THREADS_v<mmt>);
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(g);
}
*/

/*
PYBIND11_MODULE(symmul, m) {
    m.doc() = "ThunderKittens symmetric matmul module";
    BIND_FUNCTION(m, "symmul4096_4096", symmul4096_4096, globals, A, B, C);
//BIND_FUNCTION(m, "symmul4096_16384", symmul4096_16384, globals);
}
*/

////////////////////////////////////////////////////////////
// Benchmarking
////////////////////////////////////////////////////////////

template<typename mmt>
int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cudaStatus;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Block size: " << mmt::M_BLOCK*64 << "x" << mmt::N_BLOCK*64 << "\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);

    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    if(true) cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M*N*sizeof(__nv_bfloat16));

    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);

    cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device" << std::endl;

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch kernel
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    for(int i = 0; i < (NCU ? 0 : 2); i++) { // warmup
        matmul<mmt>(d_A, d_B, d_C, M, N, K, grid, block);
    }

    // Start timing
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = (NCU ? 1 : 10);
    for(int i = 0; i < ITERS; i++) {
        matmul<mmt>(d_A, d_B, d_C, M, N, K, grid, block);
    }
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;

    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    // Copy result back to host
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];
    cudaMemcpy(h_C_bf16, d_C, M*N*2, cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if(error > 1.0) { // large because of bf16 vs fp32 numerics
            if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        max_error = std::max(max_error, error);
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

int main() {
    int N;
    N = 1024;
    run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // N = 3072;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<3,3,8>>(N, N, N);
    // N = 4096;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // N = 6144;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<3,3,8>>(N, N, N);
    // N = 8192;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // N = 12288;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<3,3,8>>(N, N, N);
    // N = 16384;
    // run_benchmark<matmul_template<2,4,8>>(N, N, N);
    // run_benchmark<matmul_template<2,4,12>>(N, N, N);
    // run_benchmark<matmul_template<3,3,12>>(192*12, 192*11, 8192);
    // run_benchmark<matmul_template<2,4,11>>(128*22, 256* 6, 8192);
    // run_benchmark<matmul_template<2,4,1>>(128 * 132, 256, 256);
    // run_benchmark<matmul_template<2,4,1>>(128 * 133, 256, 256);
    // run_benchmark<matmul_template<2,4,1>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<2,4,8>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<2,4,12>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<2,4,128>>(16384, 16384, 16384);
    // run_benchmark<matmul_template<3,3,12>>(192*22, 192*6*2, 8192);
    // run_benchmark<matmul_template<3,3,12>>(192*22, 192*6*2, 16384);
    // run_benchmark<matmul_template<2,4,11>>(128*22*2, 256* 6*2, 8192);
    // run_benchmark<matmul_template<3,3,12>>(192*12*2, 192*11*2, 8192*2);
    // run_benchmark<matmul_template<2,4,11>>(128*22*2, 256* 6*2, 8192*2);
    return 0;
}