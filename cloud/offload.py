"""Prefill a templated prompt on Modal and restore it into the local llama-server's slot 0.
  cloud\\.venv\\Scripts\\python cloud\\offload.py PROMPT_FILE [--n-ctx 131072 --kv q4_0 --slots ..\\slots --url http://127.0.0.1:8080]
Needs `modal deploy cloud/modal_prefill.py` once. The local server must run the same build, GGUF, -c and KV types,
with --slot-save-path pointing at --slots. Prints one JSON line with the timings."""
import argparse, hashlib, json, os, time, urllib.request
import modal

A = argparse.ArgumentParser()
A.add_argument("prompt_file")
A.add_argument("--n-ctx", type=int, default=131072)
A.add_argument("--kv", default="q4_0")
A.add_argument("--slots", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "slots"))
A.add_argument("--url", default="http://127.0.0.1:8080")
a = A.parse_args()

prompt = open(a.prompt_file, encoding="utf-8").read()
name = "p" + hashlib.sha256(f"{a.n_ctx}|{a.kv}|{prompt}".encode()).hexdigest()[:16]
t0 = time.time()
P = modal.Cls.from_name("inkling-prefill", "Prefiller")(n_ctx=a.n_ctx, kv=a.kv)
remote = P.prefill_prompt.remote(prompt, name)
t1 = time.time()
vol = modal.Volume.from_name("inkling")
os.makedirs(a.slots, exist_ok=True)
with open(os.path.join(a.slots, name), "wb") as f:
    for chunk in vol.read_file(f"slots/{name}"):
        f.write(chunk)
vol.remove_file(f"slots/{name}")
t2 = time.time()
req = urllib.request.Request(a.url + "/slots/0?action=restore", json.dumps({"filename": name}).encode(),
                             {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
os.remove(os.path.join(a.slots, name))
print(json.dumps({"tokens": remote["prompt_tokens"], "modal_s": round(t1 - t0, 1), "prefill_s": remote["prefill_s"],
                  "download_s": round(t2 - t1, 1), "restore_s": round(time.time() - t2, 1), "n_restored": r.get("n_restored")}))
