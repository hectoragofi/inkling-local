"""End-to-end: long doc + needles + factual questions, through the proxy (compressed) or straight to llama-server.
  python e2e.py URL [--chars N] [--no-compress]"""
import json, random, re, string, sys, time, urllib.request

url = sys.argv[1]
chars = int(sys.argv[sys.argv.index("--chars") + 1]) if "--chars" in sys.argv else 10**9
nocomp = "--no-compress" in sys.argv
doc = open(r"..\quality\corpus.txt", encoding="utf-8").read()[:chars]
rng = random.Random(1)
codes = ["".join(rng.choices(string.ascii_uppercase + string.digits, k=8)) for _ in range(3)]
paras = doc.split("\n\n")
for i, c in enumerate(codes):
    paras.insert(int(len(paras) * (i + 1) / 4), f"Note for the reader: the secret code number {i + 1} is {c}.")
doc = "\n\n".join(paras)
QS = [
    ("List every secret code number and its value that appears in the document.", codes),
    ("According to the document, which command-line flag sets the number of layers to store in VRAM?", ["-ngl", "gpu-layers"]),
    ("According to the document, what is the default port of llama-server?", ["8080"]),
]
for q, want in QS:
    body = {"model": "inkling", "max_tokens": 4000, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
            "messages": [{"role": "user", "content": doc + "\n\n" + q + " Answer briefly."}]}
    if nocomp:
        body["compress"] = False
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                                                  {"Content-Type": "application/json"}), timeout=7200).read())
    ans = r["choices"][0]["message"]["content"] or ""
    tm = r.get("timings", {})
    hit = sum(w.lower() in ans.lower() for w in want)
    print(f"Q: {q[:60]}\n  prompt_tokens={r['usage']['prompt_tokens']} prefill={tm.get('prompt_per_second', 0):.0f} tok/s "
          f"decode={tm.get('predicted_per_second', 0):.1f} wall={time.time() - t0:.0f}s hit={hit}/{len(want) if len(want) > 1 and want is codes else 1}\n"
          f"  A: {ans.strip()[:200]!r}", flush=True)
