# Inkling 2 Mini on an 8 GB laptop GPU

This repo runs and tunes Thinking Machines' Inkling 2 Mini (a 26B looped MoE) locally on an RTX 4060 Laptop with 8 GB of VRAM, a Ryzen 7 7840HS, and 32 GB of DDR5-5600.

Result: about **26 tok/s decode** and **200 tok/s prefill** at a 32K context. The starting point was 6 tok/s with naive `--n-cpu-moe` flags.

## Run

```powershell
.\start-inkling.ps1          # chat UI + OpenAI-compatible API at http://127.0.0.1:8080
.\start-inkling.ps1 -Safe    # use when the 4060 also drives a display (needs less VRAM)
```

## Setup: files not in git

- **llama.cpp fork:** `llama\` holds the unsloth llama.cpp prebuilt `b12588-mix-c0c3de8`, Windows x64 cuda13-newer build, from <https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8>.
  - Copy `cudart64_13.dll`, `cublas64_13.dll` and `cublasLt64_13.dll` into it from ggml-org `cudart-llama-bin-win-cuda-13.5-x64.zip`. The bundle ships without them, and without them CUDA silently doesn't load.
  - Mainline llama.cpp gives gibberish on these files until PR #32704 is merged.
- **Models:** take these from <https://huggingface.co/unsloth/Inkling-2-Mini-GGUF>:
  - `Inkling-2-Mini-UD-IQ3_XXS.gguf`
  - `Inkling-2-Mini-UD-Q4_K_XL.gguf`
- **Tuned model file:** `custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer.gguf`. Build it with `hillclimb\mix.py`, which takes UD-IQ3_XXS and swaps in the native NVFP4 experts of layers 0–5 and 18–19 from UD-Q4_K_XL. Tensor bytes are copied as they are; nothing is requantized.

## Layout

- `hillclimb\`: the experiment directory used with the [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill).
  - `PLAN.md`: every experiment, with numbers and verdicts.
  - `THEORY.md`: the ceilings.
  - `ledger.jsonl`, `configs\`, `logs\`.
  - Helpers: `srv.ps1`, `ceil.py`, `mix.py`, `quick.py`.
- `quality\`: the KLD measurements against UD-Q4_K_XL, using llama.cpp docs as the text corpus.

## Key findings

- **Compute buffer:** at the default `-ub 512` it takes 3.4 GB of VRAM. Use `-ub 64`–`80`.
- **Expert placement:** keep all of the looped core (layers 6–17, which run 3× per token) on the GPU, and put the experts of layers 0–5 and 18–23 on the CPU (`-ot`).
- **Loading:** `--load-mode none` gives better prefill and TTFT.
- **What didn't help:** DSpark speculative decoding and requantizing the CPU-resident experts both lose on this machine.
- **The limit:** the GPU side is bound by launch and glue overhead (~13K graph nodes per token). The practical llama.cpp ceiling here is ~35 tok/s.
