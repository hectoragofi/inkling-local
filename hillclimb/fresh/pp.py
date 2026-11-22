# prefill tok/s for an N-token cold prompt (first N tokens of doc2), 2 samples (cache_prompt off)
import json, sys, os, importlib.util
spec = importlib.util.spec_from_file_location("b", "bench120.py"); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
cfg = json.load(open(sys.argv[1])); n = int(sys.argv[2]); cfg["name"] = cfg["name"] + "-pp"
toks = json.load(open("slots/doc2.tokens.json"))[:n]
p = b.start(cfg)
try:
    res = []
    for i in range(2):
        r = b.req("/completion", {"prompt": [200002 + i] + toks, "n_predict": 1, "cache_prompt": False, "temperature": 0})
        res.append(r["timings"]["prompt_per_second"])
    print(f"PP {cfg['name']} {n}: {[round(x,1) for x in res]} tok/s")
finally:
    b.stop(p)
