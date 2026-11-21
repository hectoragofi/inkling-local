// Cold-RAM CPU MUL_MAT_ID bench: one decode token, top-8 of 128 experts, Inkling 2 Mini shapes.
//   moebench.exe [threads] [iters]
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static double bench(ggml_type t, int64_t K, int64_t M, int nth, int iters, std::mt19937 & rng) {
    const int64_t E = 128, U = 8;
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, t, K, M, E);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, 1);
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, nth);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    // random weight bytes; for scale bytes keep values small so nothing overflows
    std::vector<uint8_t> w(ggml_nbytes(as));
    for (auto & x : w) x = (uint8_t) (rng() & 0x3F);
    ggml_backend_tensor_set(as, w.data(), 0, w.size());
    std::vector<float> bv(K);
    for (auto & x : bv) x = (float) ((rng() % 2001) - 1000) / 1000.0f;
    ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(float));

    const double bytes_per = (double) ggml_nbytes(as) / E * U;
    double tot = 0;
    for (int it = 0; it < iters + 3; ++it) {
        int32_t id[8];
        for (int u = 0; u < U; ++u) {
            bool dup;
            do { id[u] = (int32_t) (rng() % E); dup = false; for (int v = 0; v < u; ++v) dup |= id[v] == id[u]; } while (dup);
        }
        ggml_backend_tensor_set(ids, id, 0, sizeof(id));
        auto t0 = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(be, gf);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (it >= 3) tot += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    const double us = tot / iters;
    printf("%-8s K=%5lld M=%5lld  %7.1f us  %5.1f MB  %5.1f GB/s\n", ggml_type_name(t), (long long) K, (long long) M, us, bytes_per / 1e6, bytes_per / us / 1e3);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return us;
}

int main(int argc, char ** argv) {
    const int nth   = argc > 1 ? atoi(argv[1]) : 8;
    const int iters = argc > 2 ? atoi(argv[2]) : 200;
    ggml_cpu_init();
    std::mt19937 rng(1);
    printf("threads %d\n", nth);
    // gate_up [2304 -> 2*1152], down [1152 -> 2304]
    bench(GGML_TYPE_NVFP4,   2304, 2304, nth, iters, rng);
    bench(GGML_TYPE_NVFP4,   1152, 2304, nth, iters, rng);
    bench(GGML_TYPE_IQ3_XXS, 2304, 2304, nth, iters, rng);
    bench(GGML_TYPE_IQ4_NL,  1152, 2304, nth, iters, rng);
    bench(GGML_TYPE_Q8_0,    2304, 2304, nth, iters, rng);
    bench(GGML_TYPE_Q4_0,    2304, 2304, nth, iters, rng);
    if (argc > 3) {
        for (ggml_type t : {GGML_TYPE_IQ3_S, GGML_TYPE_Q3_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_IQ2_S, GGML_TYPE_Q2_K}) {
            bench(t, 2304, 2304, nth, iters, rng);
        }
        return 0;
    }
    // fixed cost: tiny expert matrices
    bench(GGML_TYPE_Q8_0,      64,   64, nth, iters, rng);
    return 0;
}
