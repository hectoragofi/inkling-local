# PLAN: Inkling 2 Mini hill-climb on RTX 4060 Laptop 8 GB

Targets: c1 decode (main), prefill at 4K/16K, 32K context stable (needle at ~30K), quality = pick the best quant
that runs well. Engine: unsloth llama.cpp b12588-mix-c0c3de8 (cuda13-newer). Harness: inference-optimisation
bench.py (6 reps, temperature 1.0, reasoning on, 512 max tokens). Noise rule: candidate must beat both neighbouring
controls. Environment: AC power, Windows Balanced plan (not changed), one server at a time.

## Log

- 21 Nov. **Bundle is missing the CUDA runtime.** The unsloth cuda13-newer zip has ggml-cuda.dll but no cudart64_13 / cublas64_13, so CUDA silently didn't load and everything ran on CPU (5.8 tok/s, 3.4 tok/s prompt). Fixed by adding the NVIDIA-signed DLLs from ggml-org b12606 `cudart-llama-bin-win-cuda-13.5-x64.zip`.
- **Launcher.** Start-Process handed the caller's pipe to the child, so bench.py hung. Now launched via WMI Win32_Process.Create. PowerShell `param()` also ate llama's `-c` (prefix of `-cmd`), now raw `$args`.
- **REJECT `--n-cpu-moe 12`, `-ub 512` (my first advice).** At ub 512 the compute buffer is 3387 MiB, VRAM fills (7.8 GB) and decode is 6.3 tok/s. Even all experts on CPU (`-ncmoe 24`, ub 512) gives only 7.0 tok/s.
- **ACCEPT `-ub 64`.** Compute buffer 3387 → 582 MiB, which frees about 2.8 GB for experts.
- **Expert placement, screening single runs at ub 64:**

  | experts on CPU | decode | VRAM |
  |---|---:|---:|
  | all | 7.0 tok/s | 3.1 GB |
  | layers 0–7 + 18–23 | 13.9 tok/s | 7.3 GB |
  | layers 0–5 + 18–23 | 13.1 tok/s | 7.9 GB, at the edge |

  Keep 0–7 + 18–23.
- **CPU expert kernels are compute-bound.** CPU-only decode is 5.8 tok/s against a 15.7 bandwidth ceiling. Experts are NVFP4 (down) + IQ2/IQ3 (gate_up), with no fast CPU path, so the effective CPU expert read is about 11 GB/s versus 52.5 measured. This is the main gap to the 41 tok/s ceiling.
- **Threads, screening:**

  | -t | decode |
  |---|---:|
  | 8 (default) | 12.1 / 13.4 (two runs) |
  | 6 | 11.3 |
  | 16 | 7.6 |

  REJECT 16 and 6.
- `-ub 32` with outer-only placement: 9.3 tok/s at 7.8 GB VRAM. REJECT, near sysmem spill.
- `--load-mode none` (no mmap with CPU overrides): 15.4 tok/s, screening single run. Rigorous A/B next.
- **bench.py A/B ctl → nommap → ctl** (6 reps):

  | run | c1 decode | prefill 4K / 16K | TTFT short | RAM peak |
  |---|---:|---:|---:|---:|
  | ctl #1 | 15.19 | 152 / 154 | 887 ms | |
  | nommap | 19.87 | 165 / 166 | 534 ms | 22.4 GiB |
  | ctl #2 | 20.31 | 149 / 155 | 826 ms | 16.7 GiB |

  ctl #1 overlapped the download, so it is void for decode.
  Verdict: decode is noise. Prefill +7% beats both controls, and TTFT is much better. ACCEPT for prefill/TTFT; costs +5.6 GiB RAM.
  **Current best = iq3-nommap, ~20 tok/s c1, 165 tok/s prefill.**
- Q4_K_XL ceilings: experts on CPU for 0–9 + 18–23 → 28 tok/s (5.9 GB GPU). For 0–11 + 18–23 → 26 tok/s. Its experts are native NVFP4.
- 21 Nov ~15:45: handed the GPU to the Fable hill-climb.

## 21 Nov, Fable hill-climb (second session)

Environment change: the desktop now sits on the iGPU, so the 4060 shows 0 MiB used / 7956 MiB free (was ~7096 free). All numbers below are with that extra room; if the dGPU drives a display again, fall back to the 0–7+18–23 split.

- **Decomposition (screens, 300-token gens):** CPU expert applications 12 / 18 / 24 → 38.1 / 43.1 / 47.8 ms per token. So one expert application on CPU costs ~0.8 ms more than on GPU, and the all-GPU extrapolation is ~28 ms (35 tok/s). The GPU side (launch overhead + dense re-reads), not the CPU kernels, is most of the time.
- `GGML_CUDA_DISABLE_GRAPHS=1`: 26.2 → 11.2 tok/s. CUDA graphs are active and worth 2.3×. (srv.ps1 now takes INK_SET / INK_EXE env vars.)
- **REJECT requantizing CPU experts to Q8_0** (lever 1). Built IQ3-core + Q8_0-outer mix (llama-quantize --allow-requantize from Q4_K_XL's native NVFP4, assembled with mix.py): 21.4 tok/s vs 26.3. The CPU path is bandwidth-bound, not kernel-bound: the extra 420 MB/token cost 8.6 ms = 49 GB/s, which is the RAM bandwidth. The old "CPU kernels 4x under bandwidth" reading was wrong (CPU-only decode is slow for other reasons). Fewer bytes is the only CPU lever; faster formats are not.
- **REJECT DSpark `--spec-draft-n-max 3`** at temperature 1.0 (core split, drafter on GPU): 17 tok/s vs 23.2. Acceptance 33–38%, mean length ~2, but a 4-token verify pass costs ~117 ms vs 43 ms (CPU experts scale with tokens).
- REJECT `--poll 100` (24.0), `-t 6` (25.2), `-t 4` (24.2) vs 26.3; `--prio 2` 26.5 = noise. `-fa off` fails to load with q8_0 KV.
- **A/B (bench.py, 6 reps):**

  | run | c1 decode | prefill 4K / 16K | TTFT | RAM+VRAM peak |
  |---|---:|---:|---:|---:|
  | ctl iq3-nommap | 22.77 | 169 / 169 | 403 ms | 22.5 GiB |
  | iq3-outer | 25.92 | 203 / 197 | 405 ms | 23.3 GiB |
  | ctl iq3-nommap | 22.83 | 169 / 169 | 439 ms | 22.4 GiB |
  | iq3-outer-noop | 25.90 | 127 / 125 | 385 ms | 22.8 GiB |
  | iq3-outer | 25.98 | 203 / 199 | 410 ms | 23.2 GiB |

  **ACCEPT outer-only placement** (experts of 0–5 + 18–23 on CPU, all of the looped core on GPU): +13.7% decode, +19% prefill, VRAM 7571 MiB. Prefill is PCIe-bound: it scales with the CPU-resident expert bytes copied per ubatch.
  **REJECT `--no-op-offload`**: decode equal, prefill −37% (frees only 166 MiB).
- REJECT `GGML_CUDA_GRAPH_CHUNK_SIZE` 0 / 128 / 2048 (default 512): 25.7 / 26.1 / 26.0 vs 26.2, noise.
- REJECT DSpark `--spec-draft-n-max 1` and `2` (core split): 18.4 / 17.9 tok/s vs 23.2. Mean accepted length 1.54 / 1.81, but a 2-token verify pass costs ~84 ms vs 43 ms for one token, so any batch above 1 loses at temperature 1.0.
- REJECT f16 KV (core split): 23.5 vs 23.2 with q8_0, +250 MiB VRAM. No q8_0 KV penalty on this model; the VRAM is better spent on experts.
- **Quality (llama-perplexity --kl-divergence, 20 chunks x 512 of quality\corpus.txt, reference = UD-Q4_K_XL whose experts are the native NVFP4):**

  | file | mean KLD vs Q4_K_XL | same top-1 | PPL |
  |---|---:|---:|---:|
  | UD-Q4_K_XL (reference) | 0 | 100% | 2.574 |
  | UD-IQ3_XXS stock | 0.0192 ± 0.0007 | 96.1 ± 0.3% | 2.610 |
  | mix: IQ3 core + native NVFP4 experts for 0–5, 18–19 (from Q4_K_XL) | 0.0150 ± 0.0005 | 96.4 ± 0.3% | 2.603 |
  | mix: IQ3 core + IQ2_XXS-file experts for 0–5, 18–23 | 0.0422 ± 0.0016 | 94.8 ± 0.3% | |

- **REJECT IQ2-outer mix**: 27.0 tok/s vs 26.2 (+3%) for 2.2x the KLD.
- **A/B 2 (bench.py, 6 reps):**

  | run | c1 decode | prefill 4K / 16K | TTFT | RAM+VRAM peak |
  |---|---:|---:|---:|---:|
  | mixnv-outer-noop | 25.51 | 112 / 110 | 363 ms | 24.1 GiB |
  | iq3-outer-noop | 25.88 | 127 / 125 | 370 ms | 22.8 GiB |
  | mixnv-outer | 25.94 | 181 / 178 | 368 ms | 24.4 GiB |
  | iq3-outer | 26.38 | 205 / 201 | 346 ms | 23.2 GiB |
  | iq3-outer-ub80 | 26.48 | 225 / 217 | 369 ms | 23.4 GiB |
  | iq3-outer | 26.38 | 202 / 195 | 371 ms | 23.2 GiB |

  **NVFP4-outer mix = labelled quality trade the good way**: −1 to −2% decode, −11% prefill, KLD −22%. Lossless assembly (tensor bytes copied from the two unsloth files, no requantization).
  **ACCEPT `-ub 80`**: prefill +10% over both neighbours, decode equal, VRAM +102 MiB (7677 MiB).
- **Gates (bench.py --long):**

  | config | prompt tokens | needles | prefill | decode at depth | RAM+VRAM peak |
  |---|---:|---:|---:|---:|---:|
  | mixnv-outer-ub80 `--long 30000` | 23,496 | 3/3 | 184 | 24.1 | 25.3 GiB |
  | iq3-outer-ub80 `--long 30000` | 23,496 | 3/3 | 204 | 24.5 | 24.1 GiB |
  | mixnv-outer-ub80 `--long 38000 --reps 1` | 29,725 | 3/3 | 179 | 23.8 | 25.0 GiB |

  Smoke test (391 / Canberra) passed on every config; no LOOP flags in any run. `--long 30000` only makes a 23.5K-token prompt, so the pick was re-run at 38000 to fill the 32K window.
  Note: bench.py left the last server running after one `--long` run; stopped by hand.

## STATUS (21 Nov, plateau)

**Best = `best-config.json` (mixnv-outer-ub80):** `custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer.gguf`, `-ngl 99 -c 32768 --jinja -ctk q8_0 -ctv q8_0 --load-mode none -ub 80 -ot "blk\.([0-5]|1[89]|2[0-3])\.ffn_.*_exps\.=CPU"`.
c1 decode 26.0 tok/s (ctl 22.8), prefill 201 / 196 (ctl 169), TTFT ~0.4 s, VRAM 7.68 GB, KLD vs Q4_K_XL 0.0150 (stock IQ3 0.0192).
Speed-first alternative: stock UD-IQ3_XXS with the same flags, 26.5 tok/s, prefill 227 / 219.
Fallback if the dGPU gets desktop load again (less than ~7.9 GB free): `-ub 64` and the old `[0-7]` split.

**What limits further gains (evidence):**
- All-GPU extrapolation for this engine is ~28 ms/token (35 tok/s); we are at 38 ms. The 12 outer expert layers cost ~0.8 ms each on CPU and cannot fit in VRAM (5.0 GB).
- The CPU path is at RAM bandwidth (Q8_0 test: 49 GB/s marginal), so faster formats do not help.
- The GPU part is launch/glue-bound: 13.2K graph nodes per token under CUDA graphs, vs 15 ms of pure bandwidth. Batch > 1 passes cost ~2x, which kills speculative decoding.

**Remaining levers, ranked:**
1. Fuse the Inkling glue subgraphs (clip / control / convolve / mix) into single CUDA kernels in ggml-cuda (pattern fusion like the existing topk-moe one). Needs the CUDA toolkit (nvcc) and a ggml-cuda rebuild; exact source tarball is in `src\`. Estimated up to −8 ms/token (→ ~33 tok/s). Days of work.
2. Make batch-2..4 verify passes cheap (graphs for mul_mat_id batch > 1, grouped CPU experts); only then is DSpark worth retesting.
3. +1 outer layer on GPU (~+2%) if 450 MiB VRAM can be found; not safe at 32K today.

## 21 Nov evening: self-built fork + graph rewrite (Opus)

- **Toolchain:** CUDA 13.5 toolkit + VS2022, built the fork from source for sm_89 + Zen4 native (`build\`). Unpatched self-built = prebuilt speed (24.5 vs 24.7 tok/s screen, smoke PASS).
- **Nsight Systems decode profile** (pick config, `--cuda-graph-trace=node`), per token: ~7,450 kernels; GPU busy 23.1 ms (weight matvecs ~12.5 ms, glue ~10 ms), launch gaps 2.8 ms, CPU-expert gaps 12 × ~1.2 ms ≈ 15 ms (GPU idle). The 12 CPU layers read ~400 MB of active experts per token: 7.6 ms at 52.5 GB/s, so ~half the CPU gap is not bandwidth (all 12 gaps ~1.2 ms whether the layer is 36 MB NVFP4 or 28 MB IQ3).
- **KEEP graph rewrite** (`patches\inkling2-graph-fusion.patch`):
  - `clip`: `x*min(1, t/|x|) == t * l2_norm(x, eps=t)`. 8 kernels become 2 (×144 per token).
  - `mix`: the normalized block stack is memoized per distinct block list, instead of re-concatenating and re-normalizing every block on each of the 2 mixes per layer. Exact, because softmax-weighted mixing is order-invariant.
  - llama-bench tg128, interleaved: prebuilt 25.47 / 25.54, patched 26.20 / 26.01 (**+2.4%**).
  - KLD vs Q4_K_XL: 0.01524 ± 0.0005 (was 0.01501), same-top 96.35% (was 96.37%). Smoke PASS.
- **REJECT** `-t 16` (23.9), `-t 7` (26.0) vs `-t 8` 26.3.
- **Left:** the f32 loop controllers (`loop_*`, ~230 MB/token ≈ 1 ms) could go bf16 for ~0.5 ms. Dynamic conv taps + control epilogue need new fused ops (~1 ms). The fixed ~0.5 ms per CPU split (sync/threadpool wake, ×12) is the largest non-bandwidth item left.
- **CPU expert microbench** (`hillclimb\moebench\moebench.cpp`, cold RAM, top-8 of 128, 8 threads, Zen4). Every IQ type is compute-bound at ~25 GB/s: iq3_xxs 608–685 µs per gate_up, iq3_s 754, iq2_s 538. NVFP4 and every K-quant are RAM-bound at ~53 GB/s: nvfp4 453, q4_K 432, iq4_xs 424, q3_K 342, q2_K 262. The fixed cost per op is ~5 µs.
- **Not the cause:** CUDA graphs are not re-captured per token (the 65 instantiations are first captures; `GGML_CUDA_GRAPH_DEBUG` names the node that changed). `GGML_CUDA_GRAPH_CHUNK_SIZE` 64–1024 and 0 are flat (25.9–26.3). The "0.5 ms unexplained per CPU split" was mostly node-trace inflation.
- **REJECT** gate_up of layers 20–23 requantized Q8_0 → Q4_K (faster in the microbench, higher quality, +490 MB RAM): 25.52 / 25.51 vs 25.42 / 25.18, which is noise.

## 21 Nov night: realugbun fork + Modal prefill
- **Fork:** `realugbun/llama.cpp` `inkling2-speed` (7669d89) runs our pick after `to_mainline.py`. Its 12 fetched `msg_query` tensors equal the root rows exactly. Results, interleaved: decode 31.0 / 31.0 vs 25.9 / 25.8 (+20%), prefill 196 vs 173. KLD 0.01520. In 128K mode: d0 tg 26.9 vs 23.4, d32K tg 23.0 vs 21.2, pp 162 vs 121.
- **Bench note:** llama-bench at `-d 131072` sizes the context at 131,648 and runs >80 min at the edge of VRAM (spill). Use `-d 120000` or a real server.
- **Modal prefill: KEEP.** On an H100, 106,674 tokens prefill in 39 s; the slot state is 1.14 GB. Restored locally in 1.4 s, 4 tokens are left to process locally and 3/3 needles come back. Decode at 107K context: 19.5 tok/s. Through the proxy, a 53K-token prompt took 106 s end to end, against ~5.5 min of local prefill.
- **Dropped Modal prefill (owner, 21 Nov):** the setup must be fully local. The offload code was removed from the repo, and the Modal app is stopped.
