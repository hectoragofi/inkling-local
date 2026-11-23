# Make corpus\chat.txt: a few chats with thinking, rendered through the chat template (reasoning included),
# for the router statistics. python -I gen_chat.py <config.json>
import json, os, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench
PROMPTS = [
    "I have a Python list of dicts with keys name and age. Write a function that returns the names of the three oldest people, and explain edge cases.",
    "Plan a 3-day trip to Lisbon for someone who likes food and old buildings but hates crowds. Keep it practical.",
    "Why does my CUDA kernel get slower when I add more registers per thread? Explain occupancy with a worked example.",
    "A train leaves at 14:05 and arrives at 17:50 with a 25 minute stop. How long is it moving? Then write a haiku about it.",
    "Review this SQL and say what is wrong: SELECT name, COUNT(*) FROM orders JOIN users ON users.id = orders.user GROUP BY 1 HAVING COUNT(*) > 2 ORDER BY name",
    "Explain to a 12 year old how a refrigerator moves heat from inside to outside.",
]
def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{bench.PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))
cfg = json.load(open(sys.argv[1])); cfg["name"] += "-genchat"
p = bench.start(cfg, os.path.join(HERE, "logs", cfg["name"] + ".server.log"))
out = []
try:
    for q in PROMPTS:
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": q}], "max_tokens": 1500, "temperature": 0.7, "seed": 1})
        m = r["choices"][0]["message"]
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": m.get("content") or "", "reasoning_content": m.get("reasoning_content") or ""}]
        t = post("/apply-template", {"messages": msgs})["prompt"]
        out.append(t)
        print(len(t), "chars, reasoning", len(m.get("reasoning_content") or ""), flush=True)
finally:
    bench.stop(p)
open(os.path.join(HERE, "corpus", "chat.txt"), "w", encoding="utf-8").write("\n".join(out))
