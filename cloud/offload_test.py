"""End-to-end test of Modal prefill + local decode.
  python offload_test.py make  OUT.json [--copies 2]   -> messages with a long doc + 3 hidden codes (codes in OUT.codes)
  python offload_test.py ask   OUT.json SLOT_NAME [--local-only]
     restores SLOT_NAME into the local server's slot 0 (unless --local-only), then sends the same messages and reports
     how many prompt tokens the local server still had to process, the speed, and whether the codes came back."""
import json, os, random, string, sys, time, urllib.request

URL = os.environ.get("INK_URL", "http://127.0.0.1:8080")


def post(path, body, t=7200):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))


if sys.argv[1] == "make":
    out = sys.argv[2]
    copies = int(sys.argv[sys.argv.index("--copies") + 1]) if "--copies" in sys.argv else 2
    doc = "\n\n".join([open(r"..\quality\corpus.txt", encoding="utf-8").read()] * copies)
    rng = random.Random(int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 7)
    codes = ["".join(rng.choices(string.ascii_uppercase + string.digits, k=8)) for _ in range(3)]
    paras = doc.split("\n\n")
    for i, c in enumerate(codes):
        paras.insert(int(len(paras) * (i + 1) / 4), f"Note for the reader: the secret code number {i + 1} is {c}.")
    q = "List every secret code number and its value that appears in the document. Answer briefly."
    msgs = [{"role": "user", "content": "\n\n".join(paras) + "\n\n" + q}]
    json.dump(msgs, open(out, "w", encoding="utf-8"))
    json.dump(codes, open(out + ".codes", "w"))
    print(f"wrote {out}: {len(msgs[0]['content'])} chars, codes {codes}")
else:
    msgs = json.load(open(sys.argv[2], encoding="utf-8")); codes = json.load(open(sys.argv[2] + ".codes"))
    if "--local-only" not in sys.argv:
        t0 = time.time(); r = post(f"/slots/0?action=restore", {"filename": sys.argv[3]})
        print(f"restore: {r} in {time.time() - t0:.1f}s")
    t0 = time.time()
    r = post("/v1/chat/completions", {"model": "inkling", "messages": msgs, "max_tokens": 4000, "temperature": 1.0,
                                      "top_p": 1.0, "top_k": 0, "min_p": 0.0, "id_slot": 0, "cache_prompt": True})
    tm = r["timings"]; ans = r["choices"][0]["message"]["content"] or ""
    print(f"prompt_tokens={r['usage']['prompt_tokens']} processed_locally={tm['prompt_n']} "
          f"prefill={tm['prompt_per_second']:.0f} tok/s decode={tm['predicted_per_second']:.1f} tok/s wall={time.time() - t0:.0f}s")
    print(f"codes {sum(c in ans for c in codes)}/3  A: {ans.strip()[:200]!r}")
