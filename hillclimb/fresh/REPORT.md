# Inkling 2 Mini, fresh hill-climb: decode at ~120K context, lossless

RTX 4060 Laptop 8 GB + 7840HS, realugbun fork (7669d89), model `custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf`.
Fully local. No weights requantized, KV stays q4_0. All files are in `hillclimb\fresh\`. Nothing was committed and no
branches were touched.

## Result

| metric | baseline (build-rb, 128K flags) | **decode-first** | prefill-preserving | ceiling (serial) |
|---|---:|---:|---:|---:|
| decode @ 106,674 real tokens (Modal slot) | 19.19 / 19.22 | **23.83 / 23.70** | 20.72 / 20.66 | ~34 |
| decode @ fake 120K (see method) | 18.8 | 23.38 | 20.34 | 34.0 |
| decode, empty context | 26.65 | **30.45** | 26.68 | 41.7 |
| prefill, 8K cold prompt | 120.6 | 74.9 (−38%) | 121.0 | — |
| KLD vs base-q4kxl (standard gate) | 0.0152 ± 0.0005 | **0.01501 ± 0.0005** | (same kernels) | |
| smoke 17×23 / Canberra | PASS | PASS | — | |

All numbers are medians of 4 greedy 128-token samples, run interleaved (control, candidate, control).

**Decode-first** (`best-config.json` → `decode_first`):
`build-fresh\bin\Release\llama-server.exe` + env `INK_FA_VAR=4324`, flags
`-ngl 99 -c 131072 --jinja -ctk q4_0 -ctv q4_0 --load-mode none -ub 16 -ot "blk\.([0-5]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU"`

**Prefill-preserving**: the same binary and env with the baseline flags (`-ub 64`, experts of 0–7 and 18–23 on the
CPU). It gets +8% at depth and leaves prefill unchanged.

Patch: `fresh-fattn-vec.patch`, against the fork at 7669d89. The source snapshot is in `src\` and the build in
`build-fresh\`.

## What gave the gains

1. **FA decode kernel (+8% at depth, lossless).** The per-head relative-position bias makes the FA mask per-head.
   Decode therefore runs the VEC kernel with one block per Q head.
   - In the stock kernel, each warp computes 32 K rows one at a time. Each row has a full 5-step warp-shuffle
     reduction.
   - The kernel uses 246 registers per thread, so only 2 blocks (8 warps) fit per SM.
   - One full-attention evaluation at 120K took 1544 µs. Its bandwidth floor is 0.63 ms. There are 8 such
     evaluations per token.
   - The fix (env `INK_FA_VAR=4324`) uses 4 threads per K row (8 rows in flight per warp) and caps registers at 128
     via `__launch_bounds__(128, 4)`. Result: 1086 µs per evaluation, and FA per token went from 13.2 to 9.4 ms.
   - The math is unchanged, only the fp32 summation order differs. On the decode path (`-ub 1`, q4_0 KV), KLD is
     0.01823 ± 0.0006 against 0.01868 ± 0.0007 for the stock kernel, which is within noise.
2. **VRAM: the `-ub 64` compute buffer is 2.55 GB at 131K context (+14% more).** CUDA0 held model 5264 MiB + KV
   1208 + compute 2550 = 9.0 GB on an 8 GB card. Windows sysmem fallback was already absorbing the excess.
   - Most of that buffer is the unfused f32 per-head bias chain (`get_rows` → permute/cont → add → cast). The graph
     allocator reserves it even though the fused RELBIAS CUDA kernel never touches it.
   - `-ub 16` cuts the buffer to 623 MiB. That freed room lets the expert layers 6 and 7 move to the GPU. They are
     looped ×3 and are IQ formats, which are compute-bound at ~25 GB/s on the CPU.
   - Result: 20.8 → 23.4 tok/s at fake 120K.
   - The cost is prefill (75 vs 121 tok/s), because ubatches under 32 tokens do not use op-offload.

## Ceilings (THEORY.md, from GGUF tensor sizes, 220 GB/s GPU / 52.5 GB/s CPU)

Bytes read per token:
- GPU dense: 2.03 GB. Attention matvecs ×48 evaluations, the F32 loop controllers, and the LM head.
- GPU experts: 0.83 GB.
- CPU experts: 0.55 GB.
- KV at 120K: 1.11 GB (8 full-attention evaluations × 120K × 1152 B).

Serial ceiling: 34.0 tok/s at 120K and 41.7 at empty context. With perfect CPU/GPU overlap it would be 53, but the
dependency chain prevents that. Best now = 70% of the serial ceiling at depth.

## Profile at fake 120K (nsys `--cuda-graph-trace=node`, decode stretch only)

| | baseline | best |
|---|---:|---:|
| ms/token (under nsys) | 54.5 | 45.0 |
| FA vec | 13.16 | 9.44 |
| Q8_0 attention matvecs | 5.00 | 5.02 |
| GPU experts (IQ3/IQ2/IQ4_NL/NVFP4) | ~4.5 | ~5.7 |
| Q5_K incl. LM head | 1.50 | 1.50 |
| unfused relbias `get_rows` | 1.03 (1 of 8 chains not fused at ub64) | 0 (fused at ub16) |
| glue (ctrl, conv, mix, norms, quantize) | ~4 | ~4 |
| **GPU idle (CPU experts + syncs)** | **22.9** | **17.1** |

The idle time comes from the CPU expert evaluations. Each one has ~0.95 ms of CPU compute, ~0.13 ms of
sync/launch, and a D2H → H2D round trip. There are 18 of them at baseline and 12 at best.

## Every experiment

| # | change | result | verdict |
|---|---|---|---|
| H1 | GQA-grouped VEC kernel (3 Q heads per block, per-column mask) | 9.5 tok/s vs 18.8; with the later fixes, 10.8–11.7 | REJECT (slower; not debugged further) |
| E2 | `__launch_bounds__(128, minb)`, minb 3 / 4 | 19.04 / 19.22 vs 18.84 | superseded by E3 |
| E3 | threads per K row / V row / minb: 4/32/1, **4/32/4**, 8/32/4, 16/32/4, 4/8/4, 8/8/4, 8/16/4 | 19.43, **20.34**, 20.19, 19.27, 20.13, 20.01, 20.20 vs 19.28 | KEEP 4/32/4 |
| E4 | `-t 6`, `-t 7`, `-t 16`, `--poll 100`, `--prio 2` | 20.22, 20.20, 16.99, 20.21, 20.42 vs 20.36 / 20.32 | REJECT (noise or worse) |
| E5 | `-ub 32`, `-ub 16` | 20.33, 20.84 | ub16 only as an enabler |
| E6 | `-ub 16` + experts 6–7 on GPU | **23.38** (real 107K: 23.8) | KEEP (decode-first) |
| E7 | E6 + layer 5 experts on GPU | 12.46 (weights spill to sysmem) | REJECT |
| — | llama-bench `-d 120000`, server prefill at 120K (ub 64), CPU-only prefill | >60 min / crawls past ~60K / 25 tok/s | measurement method replaced |

### Method note

`INK_FAKE_NKV=N` is in the patch and is env-gated. It widens the full-attention K/V view to N empty, masked cells
that FA still streams, so decode at depth can be timed without a prefill. It matched reality: 18.8 at fake 120K vs
19.2 real at 107K.

The real-depth numbers come from the 107K Modal-prefilled slot (`slots\doc2`). Its token list was parsed out of the
state-file header. Inference itself was all local.

## Where I disagree with `hillclimb\PLAN.md` / `THEORY.md`

- **Long-context attention is not at its floor.** It is 2.5× above it, because of warp-per-row K·Q and register
  starvation. The prior ~14 ms attention cost at 107K was treated as KV bandwidth. About 4 ms of it was kernel
  inefficiency, and ~4 ms more is still recoverable (FA is at 9.4 vs a ~5 ms floor).
- **"The CPU path is at RAM bandwidth"** holds for NVFP4. Their own microbench says the IQ types are compute-bound at
  ~25 GB/s. In 128K mode, layers 6–7 (IQ, ×3 loop) sat on the CPU, so they were the cheapest layers to move. The
  128K split was never revisited after the 32K split work.
- **`-ub 64` as a VRAM fix.** At 131K context its compute buffer is 2.55 GB, not the ~580 MiB seen at 32K. The
  128K-mode baseline was already 1 GB over VRAM, running on sysmem fallback.
- **Prefill at 120K** is not just PCIe-bound. Past ~60K it pages: FA prefill dequantizes the whole q4_0 K/V to f16
  temp buffers that grow with n_kv. Even a ~20-token prompt at 120K depth does this (3 × ~1.1 s in the profile),
  which hurts TTFT for every chat turn at depth.

## Next levers, ranked (estimated)

1. **Make the bias chain a single graph op** (the fused RELBIAS as a real op): about −1.6 GB of compute buffer at
   ub64. That lets the decode-first placement keep `-ub 64` prefill speed, and maybe one more outer expert layer.
   Est. +2–4% decode and +60% prefill vs decode-first. ~2 h.
2. **A purpose-built decode attention kernel** with an in-kernel relative bias (table of 1025 per head; beyond 1024
   it is constant per head), so there is no per-head mask. The K/V head is shared across its 3 Q heads. FA would go
   from 9.4 to ~5–6 ms/token, est. +8–10% at 120K. It also restores `mask->ne[2]==1`, which allows the GQA MMA path
   for prefill.
3. **Small-batch prompts at depth:** route Q ne[1] ≤ 8 to the VEC kernel instead of the f16 dequant path. This is a
   TTFT fix at depth (seconds per turn).
4. CPU expert evaluations: ~0.13 ms of sync per evaluation × 12, plus NVFP4 at ~73% of RAM bandwidth. Small.
5. **Hot-expert placement per expert, not per layer** (from Basalt, see below). Potentially the biggest lever.
6. **Token embedding in pinned RAM, gathered by the GPU** (from Basalt): frees VRAM for more GPU experts.

## Ideas from Basalt (jesdga95/basalt, read 22 Nov)

Basalt (MIT, a fork of Strata) runs Inkling 2 Mini at 613 tok/s on an RTX 5090. Its own docs (`docs/models/inkling2.md`)
say:
- 613–628 tok/s is the "struct" workload with DSpark drafting, where 97.5% of drafts are accepted.
- Plain decoding: 264 tok/s on a short chat, 173 at 131K, 108 at 400K.
- Prefill: 14,100 tok/s at 2K, 8,900 at 128K.
- Their pack is 19.7 GB: release NVFP4 experts bit for bit (14.7 GiB), Q8_0 attention and dense layers, and the
  drafter in BF16.

### Why it does not run here as-is

- **Blackwell only.** It needs sm_120 and CUDA 13. The 4060 is sm_89, with no FP4 block-scaled tensor cores. Its NVFP4
  prefill and expert path are built on those cores, so they would need a rewrite, not a port.
- **Needs the experts on the card.** When experts spill, Basalt reads them from RAM over PCIe into the GPU, one bus read
  per expert. Their own RTX 5060 Ti (16 GB, ~740 experts in RAM) gets 53–67 tok/s. On 8 GB most experts would spill,
  and laptop PCIe (~12–16 GB/s) is slower than computing them on the CPU at ~50 GB/s, which is what we do now.
- **Bandwidth.** The 5090 has ~1.8 TB/s against our measured ~220 GB/s, about 8×. Their 173 tok/s at 131K divided by
  8 is ≈ 21, and we are at 23.8. Per unit of bandwidth, plain decoding here is already on par.
- **Drafting.** DSpark needs a drafter (0.48 GiB) plus a verify pass that touches the union of experts across ~8
  tokens. With CPU experts that union is paid in RAM. Old `hillclimb/PLAN.md` data already showed drafting losing here.
  It is only worth revisiting once far more experts live on the GPU (lever 5).

### What to rebuild here (ideas only; our own code in the llama.cpp fork)

1. **Profile-driven expert placement (lever 5).** Basalt ships a routing profile (`data/profiles/inkling2-24x128.prof`)
   and puts the *least routed* experts in RAM and the hot ones on the card. llama.cpp's `-ot` places a whole layer's
   expert tensor on one device, so today we pick layers, not experts.
   - Plan: log router picks over a corpus to get a per-(layer, expert) hit count.
   - Plan: split each `ffn_*_exps` tensor into a GPU part (hot ids) and a CPU part (cold ids). In the MoE op, route each
     selected expert to whichever device holds it, and sum the two partial results.
   - The budget is the ~2 GB now spent on whole layers 6–7. If routing is skewed (e.g. the top 25% of experts take 60%
     of the hits), the same VRAM covers far more of the traffic per token.
   - Est.: unknown until the skew is measured. First step: dump router stats (cheap, CPU-only).
   - Lossless: the same weights, only where they live changes.
2. **Embedding in pinned RAM (lever 6).** Basalt keeps the 0.86 GiB token embedding in page-locked RAM and gathers one
   row per token. Check our GGUF's `token_embd` size and placement: if it sits in VRAM, moving it frees room for hot
   experts. Cost: one row copy per token, negligible.
3. **Relative bias inside the attention kernel, no per-head mask.** Basalt's attention takes the learned bias over the last
   1,024 positions directly. This is already lever 2 above, and it confirms that design.
4. **Suffix drafting (copying from an earlier repeat in the context).** Basalt reports +13% on a copy-heavy prompt.
   This is the ngram work the other session owns. Not duplicated here.
5. **Window layers as ring buffers; only full layers grow.** llama.cpp's SWA cache already does this. Nothing to gain.
6. **Prefill on tensor cores in 4,096-row chunks, staging spilled experts onto the card per chunk.** It is a prefill
   idea (TTFT), not decode, and with 8 GB the staging must stream layer by layer. Worth a look after levers 1 and 3
   restore fast prefill.

## Not verified

- No 120K needle test. The real-depth check used the existing 106,674-token slot, and only the first 160 characters
  of greedy text were compared.
- Prefill was not re-measured at long depth.
- The decode-first config's VRAM margin (~7.9 GB nominal) was not tested with the desktop on the dGPU.
