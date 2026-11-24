# Inkling 2 Mini: mainline rebuild hill-climb (23-24 Nov 2026)

Paused 24 Nov ~12:30. This file is the two report commits of the build agent, verbatim, newest first.
Build: `build-next-snap\ibg2\llama-server.exe`. Source: `src\llama.cpp-next-ib`, branch `ib-gqa` (095b70100).

## Next steps (tonight)

1. Wire the verified build and flag lines into `start-inkling.ps1` and `setup.ps1`, test, merge with the thinking-keeper branch (PR #7).
2. Decide on `--spec-type ngram-mod`: 64.9 vs 31.4 tok/s on copy-heavy prompts, but greedy output is no longer identical to no-drafting output.
3. Sign out and in so "Lock pages in memory" takes effect, then run the large-page microbench (CPU expert code runs at 36-43 GB/s, below the 52.5 GB/s RAM bandwidth).
4. Finish `patches/next-exp/wip-0005` (phase-swapped VRAM): expected ~35 tok/s at 32K with fast prefill.
5. 40 tok/s at 32K: the agent judges it not reachable losslessly; plateau 35.2 with the GPU busy 21 ms per token.

## Report commit 3707511

hillclimb/next: pause; report, WIP phase-swap patch, large-page check

The harness of this session refuses to write hillclimb/next/REPORT.md (agents must return findings as text),
so the report is in this message and in 27b573c.

Verified build: mainline c212d25ad + patches/next/0001-0013 = src/llama.cpp-next-ib branch ib-gqa
(095b70100, the worktree is left on it), server build-next-snap\ibg2\llama-server.exe.

Recommended flag lines (common: -m custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf -ngl 99 --jinja
--load-mode mlock --cache-ram 2048; no env vars):
- 32K: -c 32768 -ctk q8_0 -ctv q8_0 -ub 384 -ot "blk\.([0-5]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU"
  --moe-cache-mib 256 --moe-cache-cpu-share 0.95
- 128K: -c 131072 -ctk q4_0 -ctv q4_0 -ub 1024 -ot "blk\.([0-6]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU"
- 64K q8_0: -c 65536 -ctk q8_0 -ctv q8_0 -ub 512 + the 32K -ot and cache flags (VRAM peak 7925 MiB, edge)
- --spec-type ngram-mod is fast on repeated text but changes greedy outputs; add it only if that is acceptable.

Numbers vs build-rb (A/B/A, medians of 4, mlock + cache-ram):
- 32K: 32.73 vs 31.15/31.45 tok/s (+4.6%), prefill 419 vs 159. KLD 0.01517 (TILE batches), MoE cache at
  -ub 4 0.01522 vs 0.01535 without.
- 128K: 29.13 vs 27.32/27.29 (+6.7%), at 106.7K depth 25.12 vs 19.33/19.31 (+30%), prefill 285 vs 119
  (600-660 when the prompt comes first). GQA decode path KLD 0.01800 vs 0.01806.
- 64K q8_0: 29.7 tok/s, prefill 500.
- Best decode measured at 32K: 35.2 tok/s with -ub 16 --moe-cache-mib 850 (prefill unusable at ub 16).

Concurrent hot/cold work: upstream already runs the cached (GPU) experts of a layer while the CPU computes
the misses (prefetch of the CPU split's inputs). Removing both remaining host syncs per layer (next-exp 0004:
frozen cache, async copies of CPU results) gave +0.7% at 32K and nothing with all experts on the host; the
per-split cost there is the turnaround of the single scheduler thread, not the syncs. Next concrete step:
finish wip-0005 (phase swap), which should give about 35 tok/s with ub 384 prefill.

Large pages: moebench_lp reports AdjustTokenPrivileges error 1300, SeLockMemoryPrivilege not available
(VirtualAlloc with large pages would fail with 1314). Not measured. Side result: CPU expert evaluation
reaches 43 GB/s (NVFP4) and 36 GB/s (IQ3_XXS/IQ4_NL) with 8 threads, below the 52.5 GB/s the ceilings assume.

Queue, by expected size: phase swap (wip-0005, +7% at 32K), per-layer cache budgets from routing counts,
decode attention at depth (fa_vec is 0.3 ms/token at short context, more at 30K), small-batch prompts at
depth, pipelining (estimate: no gain without independent tokens; drafting is not output-identical).


## Report commit 27b573c

hillclimb/next: state of the work (report), experimental patches, decode-only cache sweep

The harness of this session refuses to write hillclimb/next/REPORT.md, so the report lives in this message.

Build in use (recommended): mainline c212d25ad + patches/next/0001-0013, server build-next-snap\ibg2.
- 0001-0004 fork commits, 0005 FA vec 4324, 0006-0007 indexed bias, 0008 MSVC fix, 0009 MMA indexed bias
  (INK_IB_MMA=1), 0010 partial expert copy for non-core layers, 0011 TILE for indexed-bias batches,
  0012 GQA-columns decode (INK_IB_GQA=0 disables), 0013 layout 4324 default.

Flag lines (all: -m custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf -ngl 99 --jinja
--load-mode mlock --cache-ram 2048, no env vars):
- 32K: -c 32768 -ctk q8_0 -ctv q8_0 -ub 384 -ot "blk\.([0-5]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU"
  --moe-cache-mib 256 --moe-cache-cpu-share 0.95   (32.7 tok/s, prefill 419)
- 32K decode-first: same with -ub 80 --moe-cache-mib 384   (34.0 tok/s, prefill 158)
- 128K: -c 131072 -ctk q4_0 -ctv q4_0 -ub 1024 -ot "blk\.([0-6]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU"
  (29.1 tok/s, 25.1 at 106.7K depth, prefill 285-660)
- 64K q8_0: -c 65536 -ctk q8_0 -ctv q8_0 -ub 512 + the 32K -ot and cache flags (29.7 tok/s, VRAM edge)

ngram-mod: greedy outputs are NOT identical with it. Once the ngram cache has drafts (second run of the
same prompt), the output differs from no drafting at character 228 (essay) and 454 (thinking): a batched
verify pass is not bit-exact with single-token decode. It fails the identical-output rule, so it is no
longer in the recommended line; it gives 64.9 vs 31.4 tok/s on a repeated thinking prompt.

Rejected, one line each:
- -t 6/7, --poll 100, --prio 2: noise.
- layers 6+7 experts on GPU at 128K: loss (layer 7 alone kept).
- F32 to BF16: 394 of 518 tensors exact, worth 1% at most.
- loop.count=1 (and so self-drafting): top-token agreement 71.9%, KLD 0.81.
- hot outer experts as IQ3: not lossless.
- embedding in pinned RAM: already on the CPU in mainline.
- bias-chain op: done upstream by the indexed bias.
- DSpark drafter: mainline cannot load it (markov_w1 201024 vs 200058, no dflare tensors); old build
  10.6/16.1 tok/s vs 27.1/27.0, acceptance 37%/73%, outputs differ.
- all experts on the host with a 4.8-5.8 GB cache: 30.5-32.4 vs 32.3; each of the 48 layer
  evaluations adds a CPU split, small GPU gaps grow from 2.4 to 8.2-8.4 ms/token.
- MoE cache sharing across loop passes and the core copy change (next-exp 0002-0003): 32.3 vs 32.7.
- LLAMA_MOE_CACHE_FREEZE (no readback of the router per lookup) and async copies of CPU results to the
  GPU (next-exp 0004): +0.7% each at 32K (34.1-34.2 vs 33.9), not additive, below the 2% step. With all
  experts on the host they remove the syncs but not the gaps (8.2 ms/token small gaps remain): the cost
  is the per-split turnaround of the single scheduler thread, not the host waits.

Decode-only cache sweep at 32K (ub 16, compute buffer 136 MiB vs 567 at ub 80):
- --moe-cache-mib 600 / 750 / 850: 35.13 / 35.05 / 35.23 tok/s (plateau), 1000: 33.0 (VRAM edge),
  1200 / 1400: 17-18 (sysmem spill). A phase swap of the compute buffer would reach about 35.2 at most
  while keeping prefill.

Ceilings (ceil_conc.py, docs routing, 32K, bandwidth only): serial today 46.9 tok/s, concurrent today
50.2, concurrent with a solved hot set 77.9; at 30K depth 42.2 / 67.8. Best measured 35.2 = 75% of
the serial ceiling.

Routing skew (share of hits of the top 10/25/50% of experts): layers 0-5 .19-.32/.42-.56/.70-.79,
core 6-17 .29-.46/.54-.71/.79-.90, 18 .53/.76/.92, 19-22 .37-.42/.64-.68/.85-.87, 23 .27/.53/.79.

