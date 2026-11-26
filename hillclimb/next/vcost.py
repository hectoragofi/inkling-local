# Cost of a k-token verify pass on a cached prefix: python -I vcost.py <config.json> [prefix_tokens]
# Sends a 512-token prefix once (cached in slot 0), then prefix + k fresh tokens for k = 1..8, 5 times each, and
# prints the median prompt_ms of the k new tokens (one ubatch). k = 1 is the cost of one decode step.
import json, os, statistics, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench
cfg = json.load(open(sys.argv[1])); cfg = dict(cfg, name=cfg["name"] + "-vcost")
n_pre = int(sys.argv[2]) if len(sys.argv) > 2 else 512
doc = bench.doc_tokens() if hasattr(bench, "doc_tokens") else None
def req(body):
    r = urllib.request.Request(f"http://127.0.0.1:{bench.PORT}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))
p = bench.start(cfg, os.path.join(HERE, "logs", cfg["name"] + ".server.log"))
res = {}
try:
    base = list(range(1000, 1000 + n_pre))  # plain token ids, fine for timing
    req({"prompt": base, "n_predict": 0, "cache_prompt": True, "id_slot": 0})
    for k in (1, 2, 3, 4, 6, 8):
        ms = []
        for rep in range(5):
            extra = [5000 + 37 * rep + i for i in range(k)]
            t = req({"prompt": base + extra, "n_predict": 0, "cache_prompt": True, "id_slot": 0})["timings"]
            if t["prompt_n"] == k: ms.append(t["prompt_ms"])
        res[k] = round(statistics.median(ms), 2) if ms else None
        print("k", k, "ms", res[k], "per token", round(res[k] / k, 2) if res[k] else None, flush=True)
finally:
    bench.stop(p)
print("VCOST " + json.dumps({"name": cfg["name"], "ms": res}))
with open(os.path.join(HERE, "ledger.jsonl"), "a") as f:
    f.write(json.dumps({"vcost": cfg["name"], "ms": res}) + "\n")
