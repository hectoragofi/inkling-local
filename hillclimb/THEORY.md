# Ceilings: Inkling 2 Mini on RTX 4060 Laptop (8 GB) + Ryzen 7 7840HS (2x16 GB DDR5-5600)

## Hardware (measured 21 Nov 2026, torch 2.14 cu126, on AC, Windows Balanced plan, model download running)
| | measured | spec |
|---|---:|---:|
| GPU copy bandwidth | 219.7 GB/s | 256 GB/s |
| GPU fp16 GEMM 4096³ / 8192³ | 28.4 / 27.9 TFLOPS | |
| CPU RAM read (torch sum) | 52.5 GB/s | 89.6 GB/s |
| PCIe host→device (pinned) | 13.3 GB/s | |
| VRAM free at idle | ~7.4 GiB (605 MiB used by desktop) | 8 GiB |

## Model (from GGUF header, UD-IQ3_XXS)
- 24 weight layers; layers 6–17 run 3× (48 layer applications). 128 experts, top-8.
- Experts 10.30 GB, dense (attn etc.) 0.77 GB, LM head 0.32 GB, embeddings 0.26 GB.
- Attention: 1 in 6 applications global (8 of 48), rest SWA window 1024. KV f16 ≈ 32 KB/token on the global layers → 1.24 GB at 32K (0.62 GB q8_0).
- Bytes read per token ≈ 3.33 GB. Dense weights are read 3× in the loop, so they count for ~1.7 GB of it.

## Decode ceilings (c1, no speculation; t = GPU bytes/220 + CPU bytes/52.5, no overlap)
| Expert placement (IQ3_XXS) | GPU resident | CPU GB/token | ceiling |
|---|---:|---:|---:|
| all GPU (does not fit) | 11.4 GB | 0 | 57 tok/s |
| outer layers 0–5,18–23 on CPU | 6.3 GB | 0.32 | 45 tok/s |
| outer + 6–7 on CPU | 5.5 GB | 0.47 | 41 tok/s |
| first 12 layers on CPU (= --n-cpu-moe 12) | ~6.3 GB | ~0.6 | ~37 tok/s |
| all experts on CPU | 1.1 GB | 1.30 | 27 tok/s |

Takeaway: an expert in the looped core costs 3× on the CPU. Offload the outer layers first.

## Prefill
- Compute: 2 × 5.4B effective active params → ~1300 tok/s at 50% of 28 TFLOPS.
- With CPU-resident experts, llama.cpp streams them over PCIe for each big ubatch: 5.3 GB / 13.3 GB/s ≈ 0.4 s per ubatch. Bigger -ub amortises it (512 → ~600 tok/s practical, 2048 → ~1000).

## Update 21 Nov (second session): measured decomposition
- dGPU now has 7956 MiB free (desktop on the iGPU), so the whole looped core fits: experts of 0–5 + 18–23 on CPU, GPU-resident 7.6 GB.
- Per token at that split: 38 ms = ~28 ms GPU + 12 CPU expert applications x ~0.8 ms. Slope measured with 12 / 18 / 24 CPU applications (38.1 / 43.1 / 47.8 ms).
- CPU experts are bandwidth-bound (49 GB/s marginal with Q8_0), not kernel-bound. The earlier "11 GB/s effective" figure came from CPU-only decode and does not describe the hybrid path.
- Practical ceiling for llama.cpp on this machine: ~35 tok/s (everything on GPU, cannot fit at IQ3). Bandwidth ceiling for the outer split stays 45 tok/s. Now 26–26.5 tok/s = 75% of practical, 58% of bandwidth.
- Prefill is PCIe-bound: op-offload copies the used CPU experts to the GPU for each ubatch, so prefill scales with CPU-resident expert bytes and with -ub (compute buffer grows 6.3 MiB per ubatch token).
