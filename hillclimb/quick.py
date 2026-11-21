"""Screening run (not a bench): python -I quick.py NAME [--keep] -- <llama-server args>
Starts server via srv.ps1, smoke test, 2x 300-token generations, prints tok/s + VRAM, stops."""
import json, subprocess, sys, time, urllib.request
name = sys.argv[1]; keep = "--keep" in sys.argv; args = sys.argv[sys.argv.index("--") + 1:]
ps = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "srv.ps1"]
def post(path, body, t=900):
    r = urllib.request.Request("http://127.0.0.1:8080" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))
def chat(prompt, n, temp=1.0, seed=None):
    b = {"model": "inkling", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": temp, "top_p": 1.0, "top_k": 0, "min_p": 0.0}
    if seed is not None: b["seed"] = seed
    return post("/v1/chat/completions", b)
subprocess.run(ps + ["stop"]); subprocess.run(ps + ["start", name] + args, check=True)
t0 = time.time(); ok = False
while time.time() - t0 < 600:
    try:
        if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3).status == 200: ok = True; break
    except Exception: time.sleep(1)
if not ok: print("NO HEALTH"); subprocess.run(ps + ["stop"]); sys.exit(1)
print("startup %.1fs" % (time.time() - t0))
try:
    r = chat("What is 17*23? Then name the capital of Australia. Be brief.", 400)
    m = r["choices"][0]["message"]; txt = (m.get("content") or "")
    print("SMOKE", "PASS" if ("391" in txt and "Canberra" in txt) else "FAIL", repr(txt[:160]))
    for i in range(2):
        r = chat("Write a detailed essay about the history of the steam engine.", 300, seed=i)
        t = r["timings"]; print("gen %.2f tok/s (%d tok)  prompt %.1f tok/s" % (t["predicted_per_second"], t["predicted_n"], t["prompt_per_second"]))
    v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print("VRAM", v)
finally:
    if not keep: subprocess.run(ps + ["stop"])
