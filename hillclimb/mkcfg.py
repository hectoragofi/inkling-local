"""Write a bench.py config: python mkcfg.py NAME -- <llama-server args>"""
import json, sys
name, args = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
ps = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "srv.ps1"]
cfg = {
    "name": name, "base_url": "http://127.0.0.1:8080", "model": "inkling",
    "launch": ps + ["start", name] + args, "stop": ps + ["stop"], "health_timeout_s": 600,
    "concurrency": 1, "prefill_lengths": [4096, 16384], "temperature": 1.0,
    "extra_body": {"top_p": 1.0, "top_k": 0, "min_p": 0.0},
    "memory": {"cap_gib": 30, "probe": ps + ["mem"], "interval_s": 3},
}
json.dump(cfg, open(f"configs/{name}.json", "w"), indent=1)
print("configs/%s.json:" % name, " ".join(args))
