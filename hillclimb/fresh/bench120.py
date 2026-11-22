# Decode-at-depth harness: llama-server + a saved ~120K-token KV slot, restored before every sample.
#   python bench120.py prep  <config.json>        prefill the document once and save slot 'doc120'
#   python bench120.py run   <config.json> [n]    n samples (default 6): restore slot, decode 128 tokens
#   python bench120.py short <config.json> [n]    same at empty context (fresh short prompt)
# config: {"name":..., "exe":..., "args":[...], "env":{...}}; results appended to ledger.jsonl
import json, os, subprocess, sys, time, urllib.request, statistics, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\hecto\models\inkling"
PORT = 8091
SLOTS = os.environ.get("BENCH_SLOTS", os.path.join(HERE, "slots"))
DOC_TOKENS = 119500
N_PRED = 128

def req(path, body=None, timeout=3600):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())

def others_running():
    out = subprocess.run(["powershell", "-NoProfile", "-c",
        "Get-Process llama-server,llama-bench,llama-perplexity -ErrorAction SilentlyContinue | % Id"],
        capture_output=True, text=True).stdout.split()
    return out

def start(cfg):
    if others_running():
        sys.exit(f"GPU busy: llama processes {others_running()}")
    os.makedirs(SLOTS, exist_ok=True)
    env = dict(os.environ); env.update(cfg.get("env", {}))
    log = open(os.path.join(HERE, "logs", cfg["name"] + ".server.log"), "w")
    exe = cfg.get("exe", os.path.join(ROOT, r"build-rb\bin\Release\llama-server.exe"))
    args = cfg.get("wrap", []) + [exe, "--port", str(PORT), "--host", "127.0.0.1", "-np", "1", "--no-webui",
            "--slot-save-path", SLOTS] + cfg["args"]
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    for _ in range(600):
        time.sleep(1)
        if p.poll() is not None:
            sys.exit(f"server exited {p.returncode}, see log")
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
    for _ in range(120):  # a wrapper (nsys) may leave the server behind for a while
        if not others_running(): return
        time.sleep(1)
    subprocess.run(["powershell", "-NoProfile", "-c", "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process"])
    time.sleep(3)

def doc_tokens():
    path = os.path.join(HERE, "slots", os.environ.get("BENCH_DOC", "doc120") + ".tokens.json")
    if os.path.exists(path):
        return json.load(open(path))
    text = open(os.path.join(ROOT, "quality", "corpus.txt"), encoding="utf-8", errors="replace").read()
    toks = req("/tokenize", {"content": text})["tokens"]
    while len(toks) < DOC_TOKENS:
        toks = toks + toks
    toks = toks[:DOC_TOKENS]
    json.dump(toks, open(path, "w"))
    return toks

SUFFIX = "\n\nQuestion: summarise the main topics of the text above in a long, detailed answer.\nAnswer:"

def one(toks, depth):
    suf = req("/tokenize", {"content": SUFFIX})["tokens"]
    body = {"prompt": toks + suf, "n_predict": N_PRED, "temperature": 0, "ignore_eos": True,
            "cache_prompt": True, "id_slot": 0, "samplers": ["top_k"], "top_k": 1}
    r = req("/completion", body)
    t = r["timings"]
    return t["predicted_per_second"], t["prompt_n"], r["content"]

def main():
    mode, cfgp = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    cfg = json.load(open(cfgp))
    p = start(cfg)
    try:
        if mode == "prep":
            toks = doc_tokens()
            t0 = time.time()
            r = req("/completion", {"prompt": toks, "n_predict": 1, "cache_prompt": True, "id_slot": 0, "temperature": 0})
            print("prefill", r["timings"]["prompt_n"], "tok in", round(time.time() - t0, 1), "s")
            print(req("/slots/0?action=save", {"filename": "doc120"}))
            return
        if mode == "smoke":
            r = req("/v1/chat/completions", {"messages": [{"role": "user", "content": "What is 17*23? Then name the capital of Australia. Be brief."}],
                                            "max_tokens": 400, "temperature": 0})
            txt = r["choices"][0]["message"].get("content") or ""
            print("SMOKE", "PASS" if ("391" in txt and "Canberra" in txt) else "FAIL", repr(txt[:200]))
            return
        toks = doc_tokens() if mode == "run" else []
        res = []
        for i in range(n):
            if mode == "run":
                req("/slots/0?action=restore", {"filename": os.environ.get("BENCH_DOC", "doc120")})

            tps, pn, content = one(toks if mode == "run" else [], mode)
            res.append(tps)
            print(f"sample {i}: {tps:.2f} tok/s (prompt_n {pn})", flush=True)
        med = statistics.median(res)
        print(f"MEDIAN {cfg['name']} {mode}: {med:.2f} tok/s  {[round(x,2) for x in res]}")
        print("text:", repr(content[:160]))
        with open(os.path.join(HERE, "ledger.jsonl"), "a") as f:
            f.write(json.dumps({"ts": datetime.datetime.now().isoformat(timespec="seconds"), "name": cfg["name"],
                                "mode": mode, "median": round(med, 3), "samples": res, "text": content[:200]}) + "\n")
    finally:
        stop(p)

if __name__ == "__main__":
    main()
