"""Convert an unsloth looped-layout Inkling 2 GGUF (root files, mix.py picks) to mainline's inkling2_loop layout.
  python -I to_mainline.py IN.gguf OUT.gguf MAINLINE_HEADER.bin MAINLINE_URL
Tensor bytes are copied from IN (so custom mixes keep their types). Metadata comes from the mainline header
(first ~16 MB of any Mainline_Support/*.gguf_file). The per-layer loop_msg_query tensors (2304x4 f32) are fetched
from MAINLINE_URL by HTTP range and checked against the rows of IN's global loop_(kv_)res_query."""
import re, struct, subprocess, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdump import rd, rstr, rval

def parse(path):
    f = open(path, "rb"); assert f.read(4) == b"GGUF"
    ver = rd(f, "I"); nt, nkv = rd(f, "Q"), rd(f, "Q"); kv0 = f.tell(); align = 32
    for _ in range(nkv):
        k = rstr(f); v = rval(f, rd(f, "I"))
        if k == "general.alignment": align = v
    kv1 = f.tell(); ts = []
    for _ in range(nt):
        name = rstr(f); dims = [rd(f, "Q") for _ in range(rd(f, "I"))]; tt = rd(f, "I"); off = rd(f, "Q")
        ts.append(dict(name=name, dims=dims, type=tt, off=off))
    base = (f.tell() + align - 1) // align * align
    f.seek(kv0); kv = f.read(kv1 - kv0); f.close()
    return ver, nkv, kv, ts, align, base

src, out, mhead, url = sys.argv[1:5]
_, _, _, its, _, ibase = parse(src)
end = os.path.getsize(src) - ibase
so = sorted(its, key=lambda t: t["off"])
for a, b in zip(so, so[1:] + [dict(off=end)]): a["size"] = b["off"] - a["off"]
I = {t["name"]: t for t in its}
mver, mnkv, mkv, mts, malign, mbase = parse(mhead)
M = {t["name"]: t for t in mts}

MOD = {"attn": "q", "k": "k", "v": "v", "mlp": "ffn"}
def rename(n):
    m = re.match(r"^(blk\.\d+)\.loop_(attn|k|v|mlp)_(down|gain|forget)\.weight$", n)
    return f"{m[1]}.loop_{MOD[m[2]]}_ctrl_{m[3]}.weight" if m else n

def raw(t):
    with open(src, "rb") as f:
        f.seek(ibase + t["off"]); return f.read(t["size"])

plan = []  # (name, dims, type, bytes-or-callable)
for t in its:
    if t["name"] in ("loop_res_query.weight", "loop_kv_res_query.weight"):
        continue
    n = rename(t["name"])
    assert n in M, f"no mainline tensor for {t['name']} -> {n}"
    assert M[n]["dims"] == t["dims"], n
    plan.append((n, t["dims"], t["type"], t))

# loop_msg_query: fetch from mainline, verify against IN's global queries
rq = np.frombuffer(raw(I["loop_res_query.weight"]), np.float32).reshape(-1, 2304)
kq = np.frombuffer(raw(I["loop_kv_res_query.weight"]), np.float32).reshape(-1, 2304)
for n, mt in M.items():
    if not n.endswith("loop_msg_query.weight"):
        continue
    nb = 2304 * 4 * 4; a = mbase + mt["off"]
    b = subprocess.run(["curl.exe" if os.name == "nt" else "curl", "-sL", "-r", f"{a}-{a + nb - 1}", url], capture_output=True, check=True).stdout
    assert len(b) == nb, (n, len(b))
    q = np.frombuffer(b, np.float32).reshape(4, 2304)
    il = int(n.split(".")[1]); r = 2 * (il - 6)
    cand = {"attn": rq[r], "mlp": rq[r + 1], "k": kq[r], "v": kq[r + 1]}
    order = [next(k for k, v in cand.items() if np.array_equal(v, q[j])) for j in range(4)]
    print(n, "rows =", order)
    plan.append((n, mt["dims"], mt["type"], b))

assert {p[0] for p in plan} == set(M), (set(M) - {p[0] for p in plan}, {p[0] for p in plan} - set(M))

def pad(x, a): return (x + a - 1) // a * a
infos = b""; off = 0; offs = []
for n, dims, tt, d in plan:
    sz = d["size"] if isinstance(d, dict) else len(d)
    offs.append(off)
    nb = n.encode()
    infos += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", x) for x in dims)
    infos += struct.pack("<IQ", tt, off)
    off = pad(off + sz, malign)
with open(out, "wb") as f:
    f.write(b"GGUF" + struct.pack("<IQQ", mver, len(plan), mnkv) + mkv + infos)
    f.write(b"\0" * (pad(f.tell(), malign) - f.tell()))
    base = f.tell()
    for (n, dims, tt, d), o in zip(plan, offs):
        f.write(b"\0" * (base + o - f.tell()))
        f.write(raw(d) if isinstance(d, dict) else d)
print(f"wrote {out} {os.path.getsize(out) / 1e9:.2f} GB, {len(plan)} tensors")
