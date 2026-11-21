"""Prefill long prompts on a Modal H100, then restore the KV state on the laptop and decode locally.

  modal run cloud/modal_prefill.py::prepare            # once: rebuild the pick on the volume, prints sha256
  modal run cloud/modal_prefill.py::prefill --messages-file m.json --name doc1 [--n-ctx 131072 --kv q4_0]
  modal volume get inkling slots/doc1 slots/doc1        # then POST /slots/0?action=restore {"filename": "doc1"}

The Modal server must match the local one exactly: same build (realugbun inkling2-speed @ FORK_SHA), same GGUF
(mainline-layout pick), same -c and KV types, -np 1. Only then is a saved slot valid locally.
"""
import json, os, subprocess, time, urllib.request
import modal

FORK_SHA = "7669d89977ff6d9d48500951ce4b6a0a5d51ab21"
HF = "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main"
PICK = "/vol/pick-mainline.gguf"
HERE = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "cmake", "build-essential", "curl", "libgomp1")
    .pip_install("numpy", "huggingface_hub[hf_xet]")
    .run_commands(
        f"git init /llama && cd /llama && git remote add o https://github.com/realugbun/llama.cpp.git"
        f" && git fetch --depth 1 o {FORK_SHA} && git checkout FETCH_HEAD",
        "cd /llama && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF"
        " -DLLAMA_BUILD_TESTS=OFF -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined"
        " && cmake --build build --config Release -j 16 --target llama-server",
    )
    .add_local_file(os.path.join(HERE, "..", "hillclimb", "gdump.py"), "/tools/gdump.py")
    .add_local_file(os.path.join(HERE, "..", "hillclimb", "mix.py"), "/tools/mix.py")
    .add_local_file(os.path.join(HERE, "..", "hillclimb", "to_mainline.py"), "/tools/to_mainline.py")
)
vol = modal.Volume.from_name("inkling", create_if_missing=True)
app = modal.App("inkling-prefill", image=image)


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


@app.function(volumes={"/vol": vol}, timeout=3600, cpu=8, memory=16384)
def prepare():
    """Rebuild custom/Inkling-2-Mini-mix-IQ3core-NVFP4outer-mainline.gguf on the volume, byte for byte."""
    os.makedirs("/vol/hf", exist_ok=True)
    for f in ("Inkling-2-Mini-UD-IQ3_XXS.gguf", "Inkling-2-Mini-UD-Q4_K_XL.gguf"):
        if not os.path.exists(f"/vol/hf/{f}"):
            sh(f"curl -sL -o /vol/hf/{f} {HF}/{f}")
    sh("cd /tools && python -I mix.py /tmp/pick-root.gguf /vol/hf/Inkling-2-Mini-UD-IQ3_XXS.gguf"
       " /vol/hf/Inkling-2-Mini-UD-Q4_K_XL.gguf 'blk\\.([0-5]|1[89])\\.ffn_gate_up_exps\\.weight$'")
    url = f"{HF}/Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file"
    sh(f"curl -sL -r 0-16777215 -o /tmp/head.bin {url}")
    sh(f"cd /tools && python -I to_mainline.py /tmp/pick-root.gguf {PICK} /tmp/head.bin {url}")
    sha = subprocess.run(f"sha256sum {PICK}", shell=True, capture_output=True, text=True).stdout.split()[0]
    sh("rm -rf /vol/hf /tmp/pick-root.gguf")
    vol.commit()
    print("sha256", sha)
    return sha


@app.cls(gpu="H100", volumes={"/vol": vol}, timeout=3600, scaledown_window=120)
class Prefiller:
    n_ctx: int = modal.parameter(default=131072)
    kv: str = modal.parameter(default="q4_0")

    @modal.enter()
    def start(self):
        os.makedirs("/vol/slots", exist_ok=True)
        t0 = time.time()
        self.p = subprocess.Popen(
            ["/llama/build/bin/llama-server", "-m", PICK, "--host", "127.0.0.1", "--port", "8080", "-np", "1",
             "-ngl", "99", "-c", str(self.n_ctx), "--jinja", "-ctk", self.kv, "-ctv", self.kv,
             "-b", "2048", "-ub", "2048", "--slot-save-path", "/vol/slots", "--no-webui"])
        while True:
            try:
                if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2).status == 200:
                    break
            except Exception:
                if self.p.poll() is not None:
                    raise RuntimeError("llama-server exited")
                time.sleep(1)
        print(f"server up in {time.time() - t0:.0f}s", flush=True)

    def post(self, path, body):
        r = urllib.request.Request("http://127.0.0.1:8080" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(r, timeout=3600))

    @modal.method()
    def prefill(self, messages, name):
        return self.prefill_prompt.local(self.post("/apply-template", {"messages": messages})["prompt"], name)

    @modal.method()
    def prefill_prompt(self, prompt, name):
        """prompt = the exact templated text the local server will see (from its /apply-template)."""
        t0 = time.time()
        r = self.post("/completion", {"prompt": prompt, "n_predict": 1, "cache_prompt": True, "id_slot": 0,
                                      "temperature": 0.0})
        t1 = time.time()
        s = self.post("/slots/0?action=save", {"filename": name})
        vol.commit()
        size = os.path.getsize(f"/vol/slots/{name}")
        out = {"prompt_tokens": r["timings"]["prompt_n"], "prefill_tok_s": r["timings"]["prompt_per_second"],
               "prefill_s": round(t1 - t0, 1), "save": s, "state_mb": round(size / 1e6, 1)}
        print(out, flush=True)
        return out


@app.local_entrypoint()
def main(messages_file: str, name: str, n_ctx: int = 131072, kv: str = "q4_0"):
    msgs = json.load(open(messages_file, encoding="utf-8"))
    print(Prefiller(n_ctx=n_ctx, kv=kv).prefill.remote(msgs, name))
