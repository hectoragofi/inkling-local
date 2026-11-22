# PLAN (fresh pass) — Inkling 2 Mini decode at ~120K, lossless

Target: max decode tok/s at ~120K context, 8 GB RTX 4060 Laptop + 7840HS, fully local, zero quality loss
(KLD vs base-q4kxl within 0.0152 ± 0.0005, Canberra / 17*23 smoke test).
Baseline (given): 26.9 tok/s @0, 23.0 @32K, 19.5 @~107K (realugbun fork, build-rb).

Own source snapshot: `src/llama.cpp-rb` (copy of the fork at 7669d89, local git for diffs), build in `build-fresh/`.

## Log

- 2026-11-21 23:0x — ceilings from GGUF sizes (THEORY.md): 34.0 tok/s serial ceiling at 120K; arithmetic says
  ~9 ms/token excess in long-context attention and ~13 ms static overhead.
- 23:1x — read graph code: the per-head relative-position bias makes the FA mask per-head
  (mask->ne[2] = 24), which disables the GQA path; decode uses the VEC kernel with one block per *Q head*, so each
  K/V head (q4_0) is streamed 3x per full-attention eval (8 evals/token). Hypothesis H1: grouping the 3 Q heads of a
  K/V head into one VEC block (columns = heads, per-column mask row) cuts KV traffic 3x -> ~-7 ms at 120K.
  Patch: `fattn-vec.cuh` gqa_cols mode (env GGML_CUDA_FA_VEC_GQA=0 disables, for A/B in one binary).
- 23:3x-00:5x — llama-bench -d 120000 and server prefill at 120K are pathological: ub64 prefill ~100 tok/s then crawls
  past ~60K (VRAM full -> WDDM sysmem fallback; FA prefill f16 K/V temp buffers grow with n_kv). CPU-only experts
  prefill = 25 tok/s. Built a measurement aid instead: `INK_FAKE_NKV=N` widens the full-attention K/V view to N
  cells (empty, masked, but streamed by FA) -> decode-at-depth timing without prefill. Check: baseline at fake 120K
  = 18.8 tok/s vs given real 19.5 @107K (consistent).
- H1 GQA-grouped VEC kernel (3 Q heads per block, per-column mask row): REJECTED. 9.5 tok/s (2x slower), still slower
  after the other fixes (10.8-11.7). Not debugged further.
- nsys @fake120K (base): FA vec = 13.2 ms/token, full-attn eval 1544 us each (8/token), grid 1x2x24 = 48 blocks,
  246 regs/thread -> 2 blocks/SM. K.Q: one warp per K row, 5-step shuffle per row.
- E2 `__launch_bounds__(128, minb)`: minb 3: 19.04, minb 4 (<=128 regs): 19.22 vs 18.84 control (+2%), same text.
- E3 threads per K row / V row / minb (INK_FA_VAR=kq*1000+v*10+minb), 4 samples each, controls 19.28/19.29 (minb4):
  4/32/1 19.43, **4/32/4 20.34**, 8/32/4 20.19, 16/32/4 19.27, 4/8/4 20.13, 8/8/4 20.01, 8/16/4 20.20.
  -> KEEP 4/32/4 (+8% vs base 18.8). Full-attn eval 1544 -> 1086 us. Greedy text diverges on a near-tie (reduction
  order): validated with decode-path KLD below.
- E4 CPU levers on top (-t 6/7/16, --poll 100, --prio 2): 20.22/20.20/16.99/20.21/20.42 vs 20.36/20.32 -> noise
  or worse, REJECTED.
- nsys gaps: per CPU-expert eval ~0.95 ms compute + ~0.13 ms sync/launch, 18 evals -> ~20 ms/token idle GPU.
  One unfused relbias chain per token (sgemm, transpose, get_rows 1.0 ms, casts) — the other 7 are fused.
- VRAM: CUDA0 model 5264 + KV 1208 + compute 2550 MiB (ub64) = 9.0 GB on 8 GB -> sysmem fallback already in use.
  Compute buffer: ub32 1756, ub16 623 MiB. ~2 GB of it is the unfused f32 per-head bias chain that the fused CUDA
  RELBIAS kernel never uses (allocated anyway).
- E5 -ub 32: 20.33 (=), -ub 16: 20.84/20.72 (+2%).
- E6 -ub 16 + experts 6,7 on GPU (CPU 0-5,18-23): **23.38** (+12% over ub16, +24% over base). Also layer 5:
  12.46 (model 6590 MiB -> weights spill to sysmem) REJECTED.
- Gates: decode-path KLD (q4_0 KV, -ub 1 so every token goes through the VEC kernel): var4324 0.01823 +- 0.00060
  vs stock kernel 0.01868 +- 0.00067 -> within noise. Standard gate (best flags, -ub 16, default KV/batching):
  0.01501 +- 0.00050 (ref 0.0152) PASS, same-top 96.45%. Smoke 391/Canberra PASS.
- Real depth: restored the 106,674-token Modal slot (`slots\doc2`, tokens parsed from the state-file header), 4
  samples x 128 tokens greedy, interleaved: base-rb 19.19 / 19.19, best 23.83 / 23.70, kernel-only 20.72 / 20.66
  (base-rb 19.22 between). Greedy text identical to base for the first 160 chars.
- Empty ctx: base 26.68 / 26.63, best 30.36 / 30.54, kernel-only 26.68. Prefill 8K cold: base 120.6, kernel-only
  121.0, best 74.9 (-38%, -ub 16).
- Profile best @fake120K (nsys, 45.0 ms/token under the profiler): kernels 27.9 (FA 9.4, Q8_0 attention matvecs
  5.0, GPU experts ~5.7, Q5_K/LM head 1.5, glue ~4), GPU idle 17.1 (12 CPU NVFP4 expert evals ~1.2 ms + gaps).
  Also seen: a ~20-token prompt at 120K depth dequantizes the whole q4_0 K/V to f16 for the MMA kernel, 3x ~1.1 s
  (temp buffers land in sysmem) -> TTFT issue at depth, not decode.

## STATUS (22 Nov ~02:30)
Best decode @~107-120K: 19.2 -> 23.8 tok/s real (+24%), lossless. Plateau not reached; next levers in REPORT.md.
