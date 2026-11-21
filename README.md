# Inkling 2 Mini on an 8 GB laptop GPU

This repo runs and tunes Thinking Machines' Inkling 2 Mini (a 26B looped MoE) locally on an RTX 4060 Laptop with 8 GB of VRAM, a Ryzen 7 7840HS, and 32 GB of DDR5-5600.

Result: about **26 tok/s decode** and **200 tok/s prefill** at a 32K context. The starting point was 6 tok/s with naive `--n-cpu-moe` flags.

## Run

```powershell
.\start-inkling.ps1          # chat UI + OpenAI-compatible API at http://127.0.0.1:8080
.\start-inkling.ps1 -Safe    # use when the 4060 also drives a display (needs less VRAM)
```

```powershell
.\start-inkling.ps1 -Long       # 128K context (q4_0 KV): ~20.6 tok/s
.\start-inkling.ps1 -Compress   # also start the compressing proxy; point apps at http://127.0.0.1:8081/v1
```

## Compressing proxy (`compress\`)

`compress\proxy.py` is an OpenAI-compatible proxy that sits in front of llama-server. It compresses long messages before the model reads them:

- **Prose:** compressed with LLMLingua-2 (`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`). It runs on the CPU at ~1,460 tok/s and uses no VRAM.
- **Code:** only comments and blank lines are stripped.
- **The question:** the last paragraph of your final message is never compressed.
- **Caching:** text is cached per paragraph chunk, so a shared document compresses once and llama-server's prompt cache still hits.
- **Per request:** add `"compress": false` or `"compress_rate": 0.5` to the request body.

Setup:

```powershell
cd compress
python -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install llmlingua
```

Measured with `compress\e2e.py` on llama.cpp docs:

| Input | Tokens | 3 hidden facts | Flag question |
|---|---:|---|---|
| ~45K-token doc, compressed | 26.1K (fits 32K) | 3/3 | correct |
| 90K-char slice, raw | 30.1K | 3/3 | correct |
| same slice, compressed | 12.3K (2.45×) | no answer (thinking hit 4K tokens) | correct |

Compression cost: ~37 s of CPU for 130K words, cached afterwards.

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

## Building the fork from source (+2.4% decode)

`patches\inkling2-graph-fusion.patch` rewrites two Inkling graph helpers to launch fewer kernels:
- **`clip`:** expressed as one `l2_norm` plus one `scale`.
- **`mix`:** the normalized block stack is memoized.

KLD is unchanged (0.0152 vs 0.0150). With the CUDA 13.5 toolkit and VS2022 installed:

```powershell
cd src\llama.cpp-b12588-mix-c0c3de8; git apply ..\..\patches\inkling2-graph-fusion.patch; cd ..\..
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
cmake -S src\llama.cpp-b12588-mix-c0c3de8 -B build -G "Visual Studio 17 2022" -A x64 -T "cuda=$env:CUDA_PATH" "-DCUDAToolkit_ROOT=$env:CUDA_PATH" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release -j 14 --target llama-server llama-bench
```

Copy the three CUDA DLLs into `build\bin\Release` as for the prebuilt. Then set `$env:INK_EXE` for `hillclimb\srv.ps1`, or point `start-inkling.ps1` at `build\bin\Release\llama-server.exe`.

## Faster build: realugbun `inkling2-speed` fork (~31 tok/s)

[realugbun/llama.cpp](https://github.com/realugbun/llama.cpp) branch `inkling2-speed` is mainline llama.cpp with fused CUDA kernels for Inkling's glue ops. Mainline uses the `inkling2_loop` GGUF layout. `hillclimb\to_mainline.py` converts our pick to that layout without touching the tensor bytes. It renames the loop controllers and fetches the 12 small per-layer `loop_msg_query` tensors from unsloth's `Mainline_Support` file; those match our rows exactly.

```powershell
git clone --depth 1 -b inkling2-speed https://github.com/realugbun/llama.cpp src\llama.cpp-rb   # tested at 7669d89
# build into build-rb\ with the same cmake flags as above, copy the 3 CUDA DLLs into build-rb\bin\Release
curl.exe -sL -r 0-16777215 -o mainline-head.bin https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file
python hillclimb\to_mainline.py custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer.gguf custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf mainline-head.bin https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file
```

`start-inkling.ps1` uses the fork and the mainline pick automatically once `build-rb\` exists.

Measured with llama-bench, pick config:

| | prebuilt / my patch | fork |
|---|---:|---:|
| decode, 32K mode | 25.9 | **31.0** |
| prefill, 32K mode | 173 | **196** |
| decode, 128K mode, empty context | 23.4 | **26.9** |
| decode, 128K mode, 32K deep | 21.2 | **23.0** |
| prefill, 128K mode, 32K deep | 121 | **162** |

Quality is unchanged: KLD vs Q4_K_XL is 0.0152 (was 0.0150), same-top 96.37%, smoke test passes. At 107K tokens of context, decode is 19.5 tok/s.
