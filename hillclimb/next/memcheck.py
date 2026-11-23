# Print the device buffer sizes a config allocates: python -I memcheck.py <config.json>  (starts with -lv 4, stops at once)
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bench
cfg = json.load(open(sys.argv[1])); cfg["args"] = cfg["args"] + ["-lv", "4"]
log = os.path.join(HERE, "logs", cfg["name"] + ".mem.log")
p = bench.start(cfg, log)
try:
    vram = bench.vram_used()
finally:
    bench.stop(p)
s = open(log, encoding="utf-8", errors="replace").read()
for l in s.splitlines():
    if re.search(r"(buffer size|KV self size|compute buffer|model buffer)", l) and "MiB" in l:
        print(l.split(" I ", 1)[-1].strip()[:150])
print("nvidia-smi used MiB", vram)
