"""Speculative-decoding A/B: python -I spec_ab.py NAME -- <llama-server args>
Starts the server (srv.ps1, INK_EXE honoured), runs a copy-heavy code edit and a free-text essay, prints decode tok/s
and draft acceptance, stops the server."""
import json, subprocess, sys, time, urllib.request

name = sys.argv[1]; args = sys.argv[sys.argv.index("--") + 1:]
ps = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "srv.ps1"]
subprocess.run(ps + ["stop"]); subprocess.run(ps + ["start", name] + args, check=True)
for _ in range(600):
    try:
        if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3).status == 200: break
    except Exception: time.sleep(1)
code = open(r"..\compress\proxy.py", encoding="utf-8").read()
TASKS = [
    ("code", f"Here is a Python file:\n\n```python\n{code}\n```\n\nReturn the complete file with the variable ARGS renamed to "
             "CONFIG everywhere. Output only the code, nothing else.", 2500, False),
    ("essay", "Write a detailed essay about the history of the steam engine.", 400, False),
    ("code+think", f"Here is a Python file:\n\n```python\n{code}\n```\n\nReturn the complete file with the function strip_code "
                   "renamed to strip_comments everywhere. Output only the code.", 4000, True),
]
try:
    for tag, prompt, n, think in TASKS:
        b = {"model": "inkling", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": float(__import__("os").environ.get("SPEC_TEMP", "1.0")),
             "top_p": 1.0, "top_k": 0, "min_p": 0.0, "seed": 1, "chat_template_kwargs": {"enable_thinking": think}}
        r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                      json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800))
        open(f"spec_out_{name}_{tag}.txt", "w", encoding="utf-8").write(r["choices"][0]["message"]["content"] or "")
        t = r["timings"]; dn = t.get("draft_n", 0); da = t.get("draft_n_accepted", 0)
        ok = "CONFIG" in (r["choices"][0]["message"]["content"] or "") if tag == "code" else True
        print(f"{name:12s} {tag:10s} decode {t['predicted_per_second']:6.2f} tok/s  n={t['predicted_n']:5d}  "
              f"draft {da}/{dn} accepted  ok={ok}", flush=True)
finally:
    subprocess.run(ps + ["stop"])
