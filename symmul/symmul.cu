#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int M_BLOCK, int N_BLOCK>
struct symmul_layout {
    using  base_tile      = st_bf<64, 64>;  // shared tile of size 64 by 64
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };  // outer product input
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };     // outer product result
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; rt_bf<16, N_BLOCK*base_tile::cols> accum_bf16; rt_bf<N_BLOCK*base_tile::cols, 16> accum_transposed; }; // register_tile of size 16 by N_BLOCK * however many columns we're doing
};
template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct symmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = symmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;  // 64 rows by 64*N_BLOCK columns
    using tall_tile = st_bf<64*N_BLOCK, 16>;  // 64*N_BLOCK rows by 16 columns
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
      // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows / (M_BLOCK*64), Cblocks = args.globals.C.cols / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,
                           (task_id%super_repeat)/SUPER_M };
                           // ^ row, col ... of the block
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

        // Skip upper triangular blocks
        /*
        if (args.common.coord.x < args.common.coord.y) {
            args.num_iters = -1;
            return;
        }
        */
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                // load the outer product input information
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                                    //                      ^ task_iter goes from 0 to however much we need to traverse the K dimension
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
            zero(args.state.accum_transposed);
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
            // Store the lower triangular part
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::warpid() == 0) {
                for(int i = 0; i < N_BLOCK; i++) {
                    // store lower triangular part
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                                {args.common.coord.x, args.common.coord.y+i});
                    tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
                }
            }

            // store the upper triangular part -- first convert to bf16 so that we can tranpose
            kittens::copy(args.state.accum_bf16, args.state.accum);
            kittens::transpose_sep(args.state.accum_transposed, args.state.accum_bf16);
            auto &c_transposed = reinterpret_cast<layout::base_tile(&)[N_BLOCK][M_BLOCK]>(args.finish.c);

            warpgroup::store(reinterpret_cast<tall_tile&>(c_transposed[warpgroup::groupid()]), args.state.accum_transposed);
            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::warpid() == 0) {
                for(int i = 0; i < N_BLOCK; i++) {
                    tma::store_async(args.globals.C, c_transposed[i][warpgroup::groupid()],
                                                {args.common.coord.y+i, args.common.coord.x});
                    tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
                }
            }

            zero(args.state.accum);
            zero(args.state.accum_transposed);
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};


// ORIGINAL MATMUL TEMPLATE //
template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
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
        }
    };
};
// END ORIGINAL MATMUL TEMPLATE //

#include <cuda_bf16.h>

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

using mmt = matmul_template<2,4,8>;
using smt = symmul_template<2,4,8>;
using mmt_globals = typename mmt::layout::globals;
using smt_globals = typename smt::layout::globals;

void matmul4096_4096(mmt_globals g) {
    const size_t N = 4096;
    const size_t M = 4096;
    const size_t K = 4096;
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(prototype::detail::NUM_THREADS_v<mmt>);

    // Absolutely need to set the dynamic shared memory here!!!
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(g);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

void symmul4096_4096(smt_globals g) {
    const size_t N = 4096;
    const size_t K = 4096;
    dim3 grid(smt::grid(N, K));
    dim3 block(prototype::detail::NUM_THREADS_v<smt>);

    // Absolutely need to set the dynamic shared memory here!!!
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<smt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    prototype::lcf::kernel<smt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(g);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

PYBIND11_MODULE(symmul, m) {
    m.doc() = "ThunderKittens symmetric matmul module";
    BIND_FUNCTION(m, "matmul4096_4096", matmul4096_4096, mmt_globals, A, B, C);
    BIND_FUNCTION(m, "symmul4096_4096", symmul4096_4096, smt_globals, A, B, C);
}