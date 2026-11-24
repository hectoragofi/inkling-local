# Speculative decoding check: python -I spec.py <config.json> [temps...]   (default temps: 0 1.0)
# Free text (an essay, no thinking) and a thinking-heavy chat, 2 runs each per temperature. Prints decode tok/s,
# drafted and accepted tokens, and writes each output to logs/spec_<name>_<task>_t<temp>_<run>.txt so greedy
# outputs can be compared between configs (identical text = identical tokens).
import json, os, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench
TASKS = [
    ("essay", "Write a detailed essay about the history of the steam engine.", 400, False),
    ("think", "A farmer has 17 sheep, buys 3 groups of 4 more, then sells a third of the flock rounded down. Each sheep "
              "gives 2.5 kg of wool a year at 9 euro per kg. What is the yearly wool income? Check your work.", 900, True),
]
cfg = json.load(open(sys.argv[1]))
temps = [float(t) for t in sys.argv[2:]] or [0.0, 1.0]
p = bench.start(cfg, os.path.join(HERE, "logs", cfg["name"] + ".spec.log"))
res = {"name": cfg["name"]}
try:
    for temp in temps:
        for tag, prompt, n, think in TASKS:
            for run in range(2):
                b = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": temp,
                     "top_p": 1.0, "top_k": 0 if temp > 0 else 1, "min_p": 0.0, "seed": 1 + run,
                     "chat_template_kwargs": {"enable_thinking": think}}
                r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{bench.PORT}/v1/chat/completions",
                              json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800))
                m = r["choices"][0]["message"]
                txt = (m.get("reasoning_content") or "") + "\n----\n" + (m.get("content") or "")
                open(os.path.join(HERE, "logs", f"spec_{cfg['name']}_{tag}_t{temp:g}_{run}.txt"), "w", encoding="utf-8").write(txt)
                t = r["timings"]; dn = t.get("draft_n", 0); da = t.get("draft_n_accepted", 0)
                key = f"{tag}_t{temp:g}"
                res.setdefault(key, []).append([round(t["predicted_per_second"], 2), t["predicted_n"], dn, da])
                print(key, run, round(t["predicted_per_second"], 2), "tok/s n", t["predicted_n"], "drafted", dn, "accepted", da, flush=True)
finally:
    bench.stop(p)
print("SPEC " + json.dumps(res))
with open(os.path.join(HERE, "ledger.jsonl"), "a") as f:
    f.write(json.dumps({"spec": res}) + "\n")
