"""Multi-turn tool-use eval for the thinking keeper.

  python quality\\agent_eval.py --base http://127.0.0.1:8080 --label dropped [--n 8] [--keep] [--effort medium]

The client behaves like a typical harness: it sends the history back WITHOUT the thinking (unless --keep).
Run it once straight at llama-server and once through keeper.py, then compare.
Two task kinds, seeded, greedy, one tool call per turn:
  ledger - look up 4 items and their suppliers (8 lookups), report the discounted total
  orders - check 3 items in 3 warehouses (9 lookups), order each from the cheapest warehouse that has enough
           stock (3 orders), report the total
(A code-breaking task was dropped: at the default effort the model thought for 4,096 tokens without guessing.)
"""
import argparse, json, random, time, urllib.request

A = argparse.ArgumentParser()
A.add_argument("--base", default="http://127.0.0.1:8080")
A.add_argument("--label", default="run")
A.add_argument("--n", type=int, default=8)
A.add_argument("--keep", action="store_true", help="client sends reasoning_content back itself")
A.add_argument("--max-tokens", type=int, default=2048)
A.add_argument("--out", default="")
A.add_argument("--effort", default="", help="reasoning_effort for every request (default: the model's own, high)")
ARGS = A.parse_args()


def chat(msgs, tools):
    body = {"model": "inkling", "messages": msgs, "tools": tools, "temperature": 0, "max_tokens": ARGS.max_tokens,
            "parallel_tool_calls": False}
    if ARGS.effort:
        body["reasoning_effort"] = ARGS.effort
    r = urllib.request.Request(ARGS.base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=1800).read())


def fn(name, desc, props):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": list(props)}}}


def loop(system, user, tools, handle, max_turns):
    """Returns (final text, turns, completion tokens, prompt tokens re-read)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    out_tok = reread = 0
    for turn in range(1, max_turns + 1):
        r = chat(msgs, tools)
        m = r["choices"][0]["message"]
        out_tok += r["usage"]["completion_tokens"]
        reread += r.get("timings", {}).get("prompt_n", 0)
        a = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("tool_calls"):
            a["tool_calls"] = m["tool_calls"]
        if ARGS.keep and m.get("reasoning_content"):
            a["reasoning_content"] = m["reasoning_content"]
        msgs.append(a)
        if not m.get("tool_calls"):
            return a["content"], turn, out_tok, reread
        for tc in m["tool_calls"]:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
                res = handle(tc["function"]["name"], args)
            except Exception as e:
                res = {"error": repr(e)}
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(res)})
    return "", max_turns, out_tok, reread


def last_number(text):
    digits = "".join(c if c.isdigit() else " " for c in text.replace(",", "")).split()
    return int(digits[-1]) if digits else None


def task_ledger(seed):
    rng = random.Random(2000 + seed)
    sup = {f"S{i}": {"region": rng.choice(["north", "south", "east", "west"]), "discount_percent": rng.choice([0, 10, 20, 25, 50])} for i in range(1, 7)}
    items = {f"I{rng.randrange(100, 999)}": {"unit_price": 20 * rng.randrange(2, 30), "quantity": rng.randrange(2, 15),
                                             "supplier": rng.choice(list(sup))} for _ in range(4)}
    want = sum(v["unit_price"] * v["quantity"] * (100 - sup[v["supplier"]]["discount_percent"]) // 100 for v in items.values())
    calls = {"n": 0}

    def handle(name, a):
        calls["n"] += 1
        table = items if name == "get_item" else sup
        return table.get(str(a.get("id")), {"error": "unknown id"})

    tools = [fn("get_item", "Look up one item: unit price, quantity ordered, supplier id.", {"id": {"type": "string"}}),
             fn("get_supplier", "Look up one supplier: region and discount percent.", {"id": {"type": "string"}})]
    text, turns, tok, rr = loop(
        "You are a careful bookkeeper. Use the tools; make exactly one lookup per turn.",
        f"Order contains items {', '.join(items)}. For each item the cost is unit_price * quantity, reduced by its "
        "supplier's discount percent. Look everything up, then reply with only the total cost of the order as a number.",
        tools, handle, 14)
    return {"kind": "ledger", "seed": seed, "ok": last_number(text) == want, "steps": calls["n"], "turns": turns, "out_tok": tok, "reread": rr}


def task_orders(seed):
    rng = random.Random(3000 + seed)
    need = {i: rng.randrange(5, 40) for i in "ABC"}
    stock, best = {}, {}
    for i in need:
        costs = rng.sample(range(3, 60), 3)                       # distinct unit costs
        for w, c in zip(("W1", "W2", "W3"), costs):
            stock[(w, i)] = {"available": need[i] + rng.randrange(1, 30), "unit_cost": c}
        cheapest = min(("W1", "W2", "W3"), key=lambda w: stock[(w, i)]["unit_cost"])
        if rng.random() < 0.6:                                     # the cheapest one is often short of stock
            stock[(cheapest, i)]["available"] = rng.randrange(0, need[i])
        ok_w = [w for w in ("W1", "W2", "W3") if stock[(w, i)]["available"] >= need[i]]
        best[i] = min(ok_w, key=lambda w: stock[(w, i)]["unit_cost"])
    want = sum(need[i] * stock[(best[i], i)]["unit_cost"] for i in need)
    placed, calls = [], {"n": 0}

    def handle(name, a):
        calls["n"] += 1
        key = (str(a.get("warehouse")), str(a.get("item")))
        if key not in stock:
            return {"error": "unknown warehouse or item"}
        if name == "get_stock":
            return stock[key]
        placed.append((key[0], key[1], int(a.get("quantity", 0))))
        return {"ordered": True}

    wi = {"warehouse": {"type": "string", "description": "W1, W2 or W3"}, "item": {"type": "string", "description": "A, B or C"}}
    tools = [fn("get_stock", "Units available and unit cost of one item in one warehouse.", wi),
             fn("place_order", "Order a quantity of one item from one warehouse.", {**wi, "quantity": {"type": "integer"}})]
    text, turns, tok, rr = loop(
        "You are a careful purchasing agent. Use the tools; make exactly one tool call per turn.",
        "We need " + ", ".join(f"{q} units of item {i}" for i, q in need.items()) + ". There are warehouses W1, W2 and W3. "
        "Check stock and cost everywhere first. Then order each item, in full, from the cheapest warehouse that has "
        "enough units of it. Finally reply with only the total cost of the three orders as a number.",
        tools, handle, 18)
    ok = sorted(placed) == sorted((best[i], i, need[i]) for i in need) and last_number(text) == want
    return {"kind": "orders", "seed": seed, "ok": ok, "steps": calls["n"], "turns": turns, "out_tok": tok, "reread": rr}


rows, t0 = [], time.time()
for s in range(ARGS.n):
    for task in (task_ledger, task_orders):
        t = time.time()
        r = task(s)
        r["secs"] = round(time.time() - t, 1)
        rows.append(r)
        print(json.dumps(r), flush=True)
for kind in ("ledger", "orders"):
    k = [r for r in rows if r["kind"] == kind]
    print(f"{ARGS.label} {kind}: solved {sum(r['ok'] for r in k)}/{len(k)}, mean steps {sum(r['steps'] for r in k)/len(k):.1f}, "
          f"output tokens {sum(r['out_tok'] for r in k)}, seconds {sum(r['secs'] for r in k):.0f}")
print(f"{ARGS.label} total: solved {sum(r['ok'] for r in rows)}/{len(rows)}, output tokens {sum(r['out_tok'] for r in rows)}, "
      f"minutes {(time.time()-t0)/60:.1f}")
if ARGS.out:
    json.dump({"label": ARGS.label, "rows": rows}, open(ARGS.out, "w"), indent=1)
