# Inkling 2 Mini on an 8 GB laptop GPU

This repo runs and tunes Thinking Machines' Inkling 2 Mini (a 26B looped MoE) locally on an RTX 4060 Laptop with 8 GB of VRAM, a Ryzen 7 7840HS, and 32 GB of DDR5-5600.

Result at a 32K context: **32.7 tok/s decode** and **419 tok/s prefill**. At 128K context: 29.1 tok/s, and still **25.1 tok/s at 107K tokens deep**. Copy-heavy coding (edits, refactors) reaches **55–90 tok/s** through ngram speculation. Quality: KLD 0.015 vs UD-Q4_K_XL. The starting point was 6 tok/s with naive `--n-cpu-moe` flags.

## Quick start

Requirements:
- Windows with an NVIDIA GPU that has 8 GB of VRAM. Tuned on an RTX 4060 Laptop (sm_89); other RTX 30/40/50 cards should work with different speeds.
- 32 GB of RAM. The outer experts live in system RAM.
- Python 3.10+, git, and ~60 GB of free disk (~31 GB after `-Clean`).
- Optional, for the faster self-built server: [CUDA Toolkit 13.x](https://developer.nvidia.com/cuda-downloads), Visual Studio 2022 with the C++ workload, and CMake.

```powershell
git clone https://github.com/hectoragofi/inkling-local
cd inkling-local
.\setup.ps1              # downloads the model, builds the tuned file and llama.cpp
.\start-inkling.ps1      # chat UI + OpenAI-compatible API at http://127.0.0.1:8080
```

`setup.ps1` builds mainline llama.cpp with the patches in `patches\next\` when the CUDA toolkit, Visual Studio and CMake are installed (~33 tok/s, 419 tok/s prefill, with ngram speculation). If they aren't, it falls back to unsloth's prebuilt llama.cpp (~26 tok/s, no speculation). It is safe to re-run, and downloads resume. An existing `build-rb\` (the older realugbun fork build, ~31 tok/s) still works.

| `setup.ps1` flag | |
|---|---|
| `-Prebuilt` | skip the build and use the prebuilt |
| `-Clean` | afterwards, delete the two ~29 GB source GGUFs the tuned file was mixed from |

| `start-inkling.ps1` flag | |
|---|---|
| `-Safe` | the GPU also drives your display (needs less VRAM) |
| `-Long` | 128K context (q4_0 KV): 29 tok/s, 25 tok/s at 107K deep |
| `-Greedy` | temperature 0 by default. Use it for coding (see below) |
| `-Compress` | also start the compressing proxy; point apps at `http://127.0.0.1:8081/v1` |
| `-Effort low` | thinking effort when the app sets none: `minimal`, `low`, `medium`, `high` (the model's default), `xhigh` |
| `-NoKeeper` | no thinking keeper; llama-server itself listens on 8080 |
| `-NoSpec` | no ngram speculation (greedy output then matches plain decoding exactly) |

The API model name is `inkling`. If you run something else in the same window, Ctrl+C stops the server.

## Thinking keeper (`keeper.py`)

Inkling is trained to see its own earlier thinking in the chat history. llama-server returns the thinking as `reasoning_content`, and it puts it back into the prompt only when the app sends that field again on the next turn. Most chat apps and agent harnesses drop it. The model then reads a history it was never trained on.

`start-inkling.ps1` therefore starts `keeper.py` on port 8080 and moves llama-server to 8079. The keeper:

- remembers the thinking of every reply (in `keeper-cache.jsonl`, so it survives restarts);
- adds it back to any assistant message that arrives without it, matched by the history before it, by tool-call id, or by the reply text;
- also accepts the other names apps use (`reasoning`, `reasoning_details`, inline `<think>` tags);
- passes everything else through unchanged: the web UI, streaming, all other endpoints.

Measured with `quality\agent_eval.py`: 12 multi-turn tool tasks, greedy, with a client that drops the thinking (results in `quality\keeper-eval\`).

| | solved | output tokens | minutes |
|---|---:|---:|---:|
| thinking dropped (straight to llama-server) | 11/12 | 13,082 | 9.7 |
| thinking restored (through the keeper) | 11/12 | 9,497 | 7.6 |
| restored, `-Effort medium` | 10/12 | 10,467 | 8.2 |

- The solve rate is the same; these tasks are too easy to separate the two on that.
- On the 13-turn tasks the keeper used fewer output tokens in all six pairs (20–67% fewer). On the 9-turn tasks it used ~10% more in five of six. With its earlier thinking in view the model re-derives less.
- Lower effort did not pay: `medium` solved one task fewer and used more tokens than the default.

It needs only Python's standard library. `GET http://127.0.0.1:8080/keeper/stats` shows how many messages it restored. Apps that already send `reasoning_content` back are left alone. The Anthropic-style `/v1/messages` endpoint is passed through without this fix.

## Memory

The server needs ~10 GB of free RAM and ~17 GB of commit. Two things broke that on this laptop:

- **Prompt cache.** llama-server keeps old conversations in RAM, up to 8 GB by default. During a 40-conversation run its private memory grew from 17 GB to 25 GB, free RAM fell to 3 GB and decode slowed several times over. The launcher now passes `--cache-ram 2048`.
- **Shared GPU memory.** A browser on the integrated GPU held 16 GB of commit that no process list shows. The model then failed to load (`unable to allocate CUDA_Host buffer`). The launcher prints a warning with the three biggest users when memory is short.

## Fast coding (ngram speculation)

Self-built servers run with `--spec-type ngram-mod` (turn it off with `-NoSpec`). When the reply repeats text that is already in the context, the drafter proposes those tokens and the model verifies several at once. Measured with `hillclimb\spec_ab.py`, in tok/s:

| workload | no speculation | ngram-mod, temp 1.0 | ngram-mod, temp 0 |
|---|---:|---:|---:|
| code edit (rewrite pasted code) | 28.3 | 47.1 | 54.6 |
| code after a thinking pass that wrote it out | 28.3 | 76.8 | **90.6** |
| essay / brand-new text | 29.5 | 28.5 | 28.8 |

To get the high numbers, set `"temperature": 0` per request or start with `-Greedy`, and give it copy-heavy work. New code has nothing to copy, so it runs at the normal speed. Free text costs nothing extra. On the new build, a repeated thinking prompt runs at 64.9 tok/s with ngram-mod vs 31.4 without.

One catch: a batched verify pass is not bit-exact with single-token decode, so greedy output with ngram-mod can drift from plain decoding after a few hundred characters (seen at characters 228 and 454; it comes from numeric near-ties, not a bug). Use `-NoSpec` if you need reproducible greedy output.

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

## What `setup.ps1` does (manual steps)

- **Tuned model file:** downloads `UD-IQ3_XXS` and `UD-Q4_K_XL`, then runs `hillclimb\mix.py custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer.gguf <IQ3> <Q4> '^blk\.([0-5]|1[89])\.ffn_.*_exps\.'`.
- **Build path:** clones ggml-org llama.cpp at `c212d25ad`, applies `patches\next\0001–0013` with `git am`, and builds it into `build-ib\` (cmake flags below, with the CUDA arch taken from your GPU). Copies in the CUDA runtime DLLs, then converts the pick with `hillclimb\to_mainline.py`.
- **Prebuilt path:** unzips the unsloth prebuilt into `llama\` and copies in the CUDA runtime DLLs.

Details:

- **llama.cpp prebuilt:** `llama\` holds the unsloth llama.cpp prebuilt `b12588-mix-c0c3de8`, Windows x64 cuda13-newer build, from <https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8>.
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

- **Compute buffer:** at the default `-ub 512` it takes 3.4 GB of VRAM on the older builds. Use `-ub 64`–`80` there. The patched mainline build fits `-ub 384` at 32K (that is where its prefill comes from).
- **Expert placement:** keep all of the looped core (layers 6–17, which run 3× per token) on the GPU, and put the experts of layers 0–5 and 18–23 on the CPU (`-ot`).
- **Loading:** `--load-mode none` gives better prefill and TTFT than mmap, but under RAM pressure Windows pages the CPU experts out and decode drops below 1 tok/s. The launcher uses `--load-mode mlock` to keep them resident (needs ~10 GB free RAM at start).
- **What didn't help:** DSpark speculative decoding (slower, and mainline can't load the drafter), self-drafting with one loop pass (72% top-token agreement), requantizing the CPU-resident experts, all experts on the host with a big GPU cache.
- **The limit:** the GPU is busy ~21 ms per token and the CPU expert code reads at 36–43 GB/s (RAM can do 52.5). Best measured is 35.2 tok/s, 75% of the bandwidth ceiling. 40 tok/s at 32K doesn't look reachable without losing accuracy.

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

## Fastest build: mainline + `patches\next` (~33 tok/s, 2.6x prefill)

`patches\next\` holds 13 patches on ggml-org llama.cpp `c212d25ad` (23 Nov 2026): the realugbun fork's Inkling commits, a faster FA vec decode kernel, the indexed relative-position bias in the FA kernels, a GQA-columns decode path, the partial expert copy for non-core layers, and an MSVC build fix. Applied, they give the tree that was measured below (checked: identical to the tested commit). The 32K line also uses upstream's MoE cache (`--moe-cache-mib 256 --moe-cache-cpu-share 0.95`), which keeps hot CPU experts on the GPU.

```powershell
git clone https://github.com/ggml-org/llama.cpp src\llama.cpp-ib; cd src\llama.cpp-ib
git checkout c212d25ad; git am ..\..\patches\next\00*.patch; cd ..\..
# build into build-ib\ with the cmake flags above, copy the 3 CUDA DLLs into build-ib\bin\Release
```

Measured A/B/A against the fork build (medians of 4, mlock + `--cache-ram 2048`, quiet machine):

| | fork (`build-rb`) | **mainline + patches** |
|---|---:|---:|
| decode, 32K mode | 31.3 | **32.7** |
| prefill, 32K mode | 159 | **419** |
| decode, 128K mode, empty context | 27.3 | **29.1** |
| decode, 128K mode, 107K deep | 19.3 | **25.1** |
| prefill, 128K mode | 119 | **285–660** |

KLD vs Q4_K_XL is unchanged (0.0152 at 32K, 0.0180 on the 128K GQA path vs 0.0181). Decode-only tuning (`-ub 16 --moe-cache-mib 850`) reaches 35.2 tok/s, but prefill becomes unusable. The full log is in `hillclimb\next\REPORT.md`; patches that did not pay are in `patches\next-exp\`.

These numbers need a quiet machine. With another heavy process using the CPU, or with Windows paging part of the CPU experts out, decode falls to the mid 20s.

## Older build: realugbun `inkling2-speed` fork (~31 tok/s)

[realugbun/llama.cpp](https://github.com/realugbun/llama.cpp) branch `inkling2-speed` is mainline llama.cpp with fused CUDA kernels for Inkling's glue ops. Mainline uses the `inkling2_loop` GGUF layout. `hillclimb\to_mainline.py` converts our pick to that layout without touching the tensor bytes. It renames the loop controllers and fetches the 12 small per-layer `loop_msg_query` tensors from unsloth's `Mainline_Support` file; those match our rows exactly.

```powershell
git clone --depth 1 -b inkling2-speed https://github.com/realugbun/llama.cpp src\llama.cpp-rb   # tested at 7669d89
# build into build-rb\ with the same cmake flags as above, copy the 3 CUDA DLLs into build-rb\bin\Release
curl.exe -sL -r 0-16777215 -o mainline-head.bin https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file
python hillclimb\to_mainline.py custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer.gguf custom\Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf mainline-head.bin https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file
```

`start-inkling.ps1` uses the fork and the mainline pick automatically once `build-rb\` exists (and `build-ib\` doesn't).

Measured with llama-bench, pick config:

| | prebuilt / my patch | fork |
|---|---:|---:|
| decode, 32K mode | 25.9 | **31.0** |
| prefill, 32K mode | 173 | **196** |
| decode, 128K mode, empty context | 23.4 | **26.9** |
| decode, 128K mode, 32K deep | 21.2 | **23.0** |
| prefill, 128K mode, 32K deep | 121 | **162** |

Quality is unchanged: KLD vs Q4_K_XL is 0.0152 (was 0.0150), same-top 96.37%, smoke test passes. At 107K tokens of context, decode is 19.5 tok/s.

## License

MIT for the scripts and notes in this repo. The model weights (Thinking Machines, via unsloth's GGUFs) and llama.cpp keep their own licenses.
