# next-build bench: one server start per config, then every measurement for that config.
#   python -I bench.py <config.json> [what ...]
#   what: short (decode at empty context), pp (cold 8K-token prefill), deep (restore the 107K slot, decode),
#         check (smoke + tool call + reasoning parse); default: short pp
# config: {"name":..., "exe":..., "args":[...], "env":{...}}. Medians of 4 greedy 128-token samples.
# Results are appended to ledger.jsonl next to this file. Refuses to start while ..\..\.gpu-busy exists.
import json, os, re, subprocess, sys, time, urllib.request, statistics, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\hecto\models\inkling"
PORT = 8091
N_PRED = 128
N_SAMPLES = 4
PP_TOKENS = 8192
SLOTS = os.path.join(ROOT, "slots")             # doc2 = 106,674-token q4_0 slot (128K mode only)
DOC2_TOKENS = os.path.join(ROOT, "hillclimb", "fresh", "slots", "doc2.tokens.json")

def req(path, body=None, timeout=3600):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())

def llama_procs():
    return subprocess.run(["powershell", "-NoProfile", "-c",
        "Get-Process llama-server,llama-bench,llama-perplexity -ErrorAction SilentlyContinue | % Id"],
        capture_output=True, text=True).stdout.split()

def vram_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])

def start(cfg, logpath):
    if os.path.exists(os.path.join(ROOT, ".gpu-busy")):
        sys.exit("GPU busy flag present")
    if llama_procs():
        sys.exit(f"GPU busy: llama processes {llama_procs()}")
    env = dict(os.environ); env.update(cfg.get("env", {}))
    log = open(logpath, "w")
    args = [cfg["exe"], "--port", str(PORT), "--host", "127.0.0.1", "-np", "1", "--no-webui", "--slot-save-path", SLOTS] + cfg["args"]
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    for _ in range(900):
        time.sleep(1)
        if p.poll() is not None:
            sys.exit(f"server exited {p.returncode}, see {logpath}")
        try:
            if req("/health").get("status") == "ok":
                return p
        except Exception:
            pass
    p.kill(); sys.exit("server start timeout")

def stop(p):
    p.terminate()
    try: p.wait(60)
    except subprocess.TimeoutExpired: p.kill()
    for _ in range(60):
        if not llama_procs(): return
        time.sleep(1)

SUFFIX = "\n\nQuestion: summarise the main topics of the text above in a long, detailed answer.\nAnswer:"

def decode(prompt_tokens):
    body = {"prompt": prompt_tokens, "n_predict": N_PRED, "temperature": 0, "ignore_eos": True,
            "cache_prompt": True, "id_slot": 0, "samplers": ["top_k"], "top_k": 1}
    r = req("/completion", body)
    decode.last_prompt_ms = r["timings"]["prompt_ms"]
    return r["timings"]["predicted_per_second"], r["timings"]["prompt_n"], r["content"]

def chat(body):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))

def check():
    out = {}
    r = chat({"messages": [{"role": "user", "content": "What is 17*23? Then name the capital of Australia. Be brief."}],
              "max_tokens": 800, "temperature": 0})
    m = r["choices"][0]["message"]; txt = m.get("content") or ""
    out["smoke"] = "391" in txt and "Canberra" in txt
    out["reasoning"] = bool(m.get("reasoning_content")) and bool(txt) and "<think" not in txt
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    r = chat({"messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
              "tools": tools, "max_tokens": 800, "temperature": 0})
    m = r["choices"][0]["message"]; tc = m.get("tool_calls") or []
    out["tool_call"] = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "Paris" in tc[0]["function"]["arguments"]
    out["smoke_text"] = txt[:120]
    return out

def log_facts(logpath):
    s = open(logpath, errors="replace").read()
    facts = {}
    for name, pat in [("compute_mib", r"CUDA0 compute buffer size\s*=\s*([\d.]+) MiB"),
                      ("model_mib", r"CUDA0 model buffer size\s*=\s*([\d.]+) MiB"),
                      ("kv_mib", r"CUDA0 KV buffer size\s*=\s*([\d.]+) MiB")]:
        m = re.findall(pat, s)
        if m: facts[name] = float(m[-1])
    return facts

def main():
    cfg = json.load(open(sys.argv[1]))
    what = sys.argv[2:] or ["short", "pp"]
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    logpath = os.path.join(HERE, "logs", cfg["name"] + ".server.log")
    p = start(cfg, logpath)
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "name": cfg["name"]}
    try:
        rec["vram_mib"] = vram_used()
        rec.update(log_facts(logpath))
        doc = json.load(open(DOC2_TOKENS))
        suf = req("/tokenize", {"content": SUFFIX})["tokens"]
        for w in what:
            if w == "short":
                res = []
                for i in range(N_SAMPLES):
                    # a fresh short prompt each time (different first token), so nothing but the prompt is cached
                    tps, pn, content = decode([200002 + i] + doc[:64] + suf)
                    res.append(tps)
                rec["short"] = round(statistics.median(res), 3); rec["short_s"] = [round(x, 2) for x in res]
                rec["short_text"] = content[:120]
            elif w == "pp":
                res = []
                for i in range(2):
                    r = req("/completion", {"prompt": [200010 + i] + doc[1000:1000 + PP_TOKENS], "n_predict": 1,
                                            "cache_prompt": False, "temperature": 0})
                    res.append(r["timings"]["prompt_per_second"])
                rec["pp"] = round(statistics.median(res), 2); rec["pp_s"] = [round(x, 1) for x in res]
            elif w == "deep":
                res = []; ttft = []
                for i in range(N_SAMPLES):
                    req("/slots/0?action=restore", {"filename": "doc2"})
                    tps, pn, content = decode(doc + suf)
                    res.append(tps); ttft.append(decode.last_prompt_ms)
                rec["deep_suffix_ms"] = round(statistics.median(ttft), 1)
                rec["deep"] = round(statistics.median(res), 3); rec["deep_s"] = [round(x, 2) for x in res]
                rec["deep_prompt_n"] = pn; rec["deep_text"] = content[:120]
            elif w == "check":
                rec["check"] = check()
            rec["vram_peak_mib"] = max(rec.get("vram_peak_mib", 0), vram_used())
            print(json.dumps({k: v for k, v in rec.items() if not k.endswith("_text")}), flush=True)
    finally:
        stop(p)
        with open(os.path.join(HERE, "ledger.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    print("RESULT", json.dumps(rec))

if __name__ == "__main__":
    main()
