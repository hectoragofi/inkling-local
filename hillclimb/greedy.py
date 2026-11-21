import json, subprocess, sys, time, urllib.request
ps = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "srv.ps1"]
args = sys.argv[2:]
subprocess.run(ps + ["stop"]); subprocess.run(ps + ["start", sys.argv[1]] + args, check=True)
for _ in range(600):
    try:
        if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3).status == 200: break
    except Exception: time.sleep(1)
def chat(p, n):
    b = {"model": "inkling", "messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": 0.0, "top_k": 1}
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    m = json.load(urllib.request.urlopen(r, timeout=900))["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")
out = [chat("What is 17*23? Then name the capital of Australia. Be brief.", 400), chat("Explain how a refrigerator works in three sentences.", 250)]
json.dump(out, open(f"logs/{sys.argv[1]}.json", "w")); print("SMOKE", "391" in out[0] and "Canberra" in out[0])
subprocess.run(ps + ["stop"])
