# THEORY (fresh pass) — Inkling 2 Mini decode ceilings, RTX 4060 Laptop + 7840HS

Written before reading `hillclimb/PLAN.md` / `hillclimb/THEORY.md`. Source: `ceil_fresh.py` (GGUF tensor sizes of
`custom/Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf`), `logs/ceil.txt`.

## Model facts that matter (from GGUF + `src/models/inkling2*.cpp`)

- 24 physical layers, core 6..17 runs 3 passes -> **48 layer evaluations per token**. Each pass has its own KV and conv
  state slot (36 + 12 = 48 slots).
- Attention: 20 SWA layers (window 1024) + 4 full layers (phys 5, 11, 17, 23). With the loop, **8 full-attention
  evaluations per token** (5, 23 once; 11, 17 three times). 8 KV heads x 128, 24 Q heads (GQA 3).
- KV per full eval per token, q4_0: 2 x 1024 x 18/32 = 1152 B. At 120K: 8 x 120K x 1152 B = **1.106 GB/token**.
- Relative-position bias per head over the whole KV (distance clamped at extent 1024, beyond = const per head) ->
  per-head f16 mask n_kv x 24 per full eval (fork's fused RELBIAS writes it in one pass): 8 x 120K x 24 x 2 B x (w+r)
  = 0.09 GB/token.
- 128 experts top-8, fused gate_up (2304 x 2304) + down. Outer layers NVFP4 (35.8 MB active/eval), core IQ2_S/IQ3_XXS
  gate_up + IQ4_NL down (~24.8 MB active/eval).
- Dense per token is large: attention projections x 48 evals (1.25 GB), F32 loop controllers (x2 passes, 0.20 GB),
  F32 routers, LM head Q5_K 0.32 GB -> **2.03 GB/token on GPU regardless of placement**.

## Ceilings (100% of measured BW: GPU 220 GB/s, CPU 52.5 GB/s)

| case | GPU ms | CPU ms | serial tok/s | perfect CPU/GPU overlap |
|---|---:|---:|---:|---:|
| baseline placement (CPU experts 0-7,18-23), ctx 0 | 13.4 | 10.5 | 41.7 | 74 |
| same, ctx 32K | 14.9 | 10.5 | 39.3 | 67 |
| same, ctx 107K | 18.3 | 10.5 | 34.7 | 55 |
| same, **ctx 120K** | **18.9** | **10.5** | **34.0** | **53** |
| all experts on GPU (needs 12.7 GB, impossible) | 16.0 | 0 | 62.7 | — |

Measured baseline: 26.9 (ctx 0) = 65% of serial ceiling; 19.5 at 107K = 56%.

## Where the gap is (arithmetic, before profiling)

- ctx 0: measured 37.2 ms vs 24.0 ms ideal -> **13 ms static overhead** (small-kernel efficiency of 48 evals x ~40
  kernels, CPU<->GPU split syncs: 18 CPU expert evals/token = 36 transfers, CPU MoE kernel efficiency).
- ctx 0 -> 107K: measured +14.1 ms vs ideal +4.9 ms -> **~9 ms excess in long-context attention**. Biggest
  context-specific lever.
- Serial dependency: GPU idles during CPU experts and vice versa. True overlap only possible by splitting the experts of
  one layer across devices (needs VRAM we do not have) — the serial ceiling (34 tok/s at 120K) is the realistic one.

## Lossless levers ranked (gap x likelihood)

1. Long-context FA decode kernel (q4_0 K/V, per-head mask, GQA 3, D=128): ~9 ms at 107K. Check split count / occupancy,
   mask traffic, whether the per-head mask path forces a slow kernel. Option: compute the bias in-kernel from the
   1025-entry table (distance >= 1024 -> constant), removing the mask read entirely.
2. Static GPU overhead (~13 ms total gap incl. CPU): profile per kernel family.
3. CPU expert path: per-eval sync + threadpool wake + kernel efficiency vs 52.5 GB/s.
4. VRAM reshuffle: anything freed (compute buffers, SWA cache size) -> one more expert layer on GPU (~0.8 ms/layer
   for a core layer, x3 evals).
