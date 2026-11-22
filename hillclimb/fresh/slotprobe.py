import json, sys, os
sys.argv = ["x", "short", "configs/best.json"]
os.environ["BENCH_SLOTS"] = r"C:\Users\hecto\models\inkling\slots"
import importlib.util
spec = importlib.util.spec_from_file_location("b", "bench120.py"); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
cfg = json.load(open("configs/best.json")); cfg["name"] = "slotprobe"
p = b.start(cfg)
try:
    print(b.req("/slots/0?action=restore", {"filename": "doc2"}))
    s = b.req("/slots")
    s0 = s[0] if isinstance(s, list) else s
    print({k: (str(v)[:200]) for k, v in s0.items()})
finally:
    b.stop(p)
