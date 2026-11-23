// Large pages vs 4 KiB pages for the CPU expert weights, same ggml MUL_MAT_ID kernel.
// One decode token, top-8 of 128 experts, Inkling 2 Mini shapes, weights spread over n_layers layers (cold RAM).
//   moebench_lp.exe [threads] [iters] [layers]
// Large pages need SeLockMemoryPrivilege ("Lock pages in memory"); without it VirtualAlloc fails with 1314 and only
// the 4 KiB numbers are printed.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static bool enable_lock_privilege() {
    HANDLE tok;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) return false;
    TOKEN_PRIVILEGES tp = {};
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    if (!LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) { CloseHandle(tok); return false; }
    AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr);
    const DWORD err = GetLastError();
    CloseHandle(tok);
    if (err != ERROR_SUCCESS) printf("AdjustTokenPrivileges: error %lu (1300 = privilege not held by this account)\n", err);
    return err == ERROR_SUCCESS;
}

struct layer_w { ggml_tensor * gu; ggml_tensor * dn; };

// returns us per layer evaluation (gate_up + down), or -1 if the allocation failed
static double run(bool large, ggml_type t_gu, ggml_type t_dn, int nth, int iters, int n_layers) {
    const int64_t E = 128, U = 8, D = 2304, F = 1152;
    ggml_init_params ip = { (size_t) (8 + 4*n_layers) * ggml_tensor_overhead() + 4*ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<layer_w> L(n_layers);
    for (auto & l : L) {
        l.gu = ggml_new_tensor_3d(ctx, t_gu, D, 2*F, E);
        l.dn = ggml_new_tensor_3d(ctx, t_dn, F, D, E);
    }
    const size_t align = 4096;
    size_t total = 0;
    for (auto & l : L) total += GGML_PAD(ggml_nbytes(l.gu), align) + GGML_PAD(ggml_nbytes(l.dn), align);

    void * base = nullptr;
    size_t alloc = total;
    if (large) {
        const size_t lp = GetLargePageMinimum();
        if (lp == 0) { printf("large pages not supported\n"); return -1; }
        alloc = GGML_PAD(total, lp);
        base = VirtualAlloc(nullptr, alloc, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (!base) {
            const DWORD err = GetLastError();
            printf("VirtualAlloc(MEM_LARGE_PAGES, %.2f GB) failed: error %lu%s\n", alloc / 1e9, err,
                   err == 1314 ? " (privilege not held: grant 'Lock pages in memory' and log on again)" :
                   err == 1450 || err == 8 ? " (not enough contiguous physical memory)" : "");
            ggml_free(ctx);
            return -1;
        }
    } else {
        base = VirtualAlloc(nullptr, alloc, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!base) { printf("VirtualAlloc failed %lu\n", GetLastError()); return -1; }
    }
    ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, alloc);
    {
        char * p = (char *) base;
        for (auto & l : L) {
            ggml_backend_tensor_alloc(buf, l.gu, p); p += GGML_PAD(ggml_nbytes(l.gu), align);
            ggml_backend_tensor_alloc(buf, l.dn, p); p += GGML_PAD(ggml_nbytes(l.dn), align);
        }
    }
    // random weight bytes, small values so no scale overflows
    std::mt19937 rng(1);
    {
        uint8_t * p = (uint8_t *) base;
        for (size_t i = 0; i < alloc; i += 4) { const uint32_t r = rng() & 0x3F3F3F3F; memcpy(p + i, &r, 4); }
    }

    // one graph per layer: gate_up then down on the same 8 experts
    ggml_context * cctx = ggml_init({ (size_t) (16*n_layers) * ggml_tensor_overhead() + ggml_graph_overhead_custom(64*n_layers, false), nullptr, true });
    ggml_tensor * x   = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, D, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(cctx, GGML_TYPE_I32, U, 1);
    ggml_cgraph * gf = ggml_new_graph_custom(cctx, 64*n_layers, false);
    for (auto & l : L) {
        ggml_tensor * h = ggml_mul_mat_id(cctx, l.gu, x, ids);                 // [2F, U, 1]
        h = ggml_view_2d(cctx, h, F, U, h->nb[1], 0);                          // first half, as the input of down
        h = ggml_cont(cctx, h);
        ggml_tensor * o = ggml_mul_mat_id(cctx, l.dn, ggml_reshape_3d(cctx, h, F, U, 1), ids);
        ggml_build_forward_expand(gf, o);
    }
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, nth);
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, be);
    std::vector<float> xv(D);
    for (auto & v : xv) v = (float) ((int) (rng() % 2001) - 1000) / 1000.0f;
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));

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
    const double us = tot / iters / n_layers;
    const double bytes = (double) (ggml_nbytes(L[0].gu) + ggml_nbytes(L[0].dn)) / E * U;
    printf("%-5s %-8s/%-8s layers %d  %7.1f us per layer eval  %5.1f MB  %5.1f GB/s\n", large ? "2MiB" : "4KiB",
           ggml_type_name(t_gu), ggml_type_name(t_dn), n_layers, us, bytes / 1e6, bytes / us / 1e3);
    ggml_backend_buffer_free(cbuf);
    ggml_backend_free(be);
    ggml_free(cctx);
    ggml_backend_buffer_free(buf);
    VirtualFree(base, 0, MEM_RELEASE);
    ggml_free(ctx);
    return us;
}

int main(int argc, char ** argv) {
    const int nth      = argc > 1 ? atoi(argv[1]) : 8;
    const int iters    = argc > 2 ? atoi(argv[2]) : 100;
    const int n_layers = argc > 3 ? atoi(argv[3]) : 6;
    ggml_cpu_init();
    printf("threads %d, iters %d, layers %d, large page size %zu\n", nth, iters, n_layers, (size_t) GetLargePageMinimum());
    const bool priv = enable_lock_privilege();
    printf("SeLockMemoryPrivilege %s\n", priv ? "enabled" : "NOT available");
    for (int rep = 0; rep < 2; ++rep) {
        run(false, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, nth, iters, n_layers);
        if (priv) run(true, GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, nth, iters, n_layers);
        run(false, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, nth, iters, n_layers);
        if (priv) run(true, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, nth, iters, n_layers);
    }
    return 0;
}
